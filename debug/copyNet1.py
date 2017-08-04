def train_finetuned_model(lr=1e-5, verbose=True):
    file_path = get_file('vgg16.h5', VGG16_WEIGHTS_PATH, cache_subdir='models')
    if verbose:
        print('Building VGG16 (no-top) model to generate bottleneck features.')

    vgg16_notop = build_vgg_16()
    vgg16_notop.load_weights(file_path)
    for _ in range(6):
        vgg16_notop.pop()
    vgg16_notop.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])    

    if verbose:
        print('Bottleneck features generation.')

    train_batches = get_batches('train', shuffle=False, class_mode=None, batch_size=BATCH_SIZE)
    train_labels = np.array([0]*1000 + [1]*1000)
    train_bottleneck = vgg16_notop.predict_generator(train_batches, steps=2000 // BATCH_SIZE)
    valid_batches = get_batches('valid', shuffle=False, class_mode=None, batch_size=BATCH_SIZE)
    valid_labels = np.array([0]*400 + [1]*400)
    valid_bottleneck = vgg16_notop.predict_generator(valid_batches, steps=800 // BATCH_SIZE)

    if verbose:
        print('Training top model on bottleneck features.')

    top_model = Sequential()
    top_model.add(Flatten(input_shape=train_bottleneck.shape[1:]))
    top_model.add(Dense(4096, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(4096, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(2, activation='softmax'))
    top_model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    top_model.fit(train_bottleneck, to_categorical(train_labels),
                  batch_size=32, epochs=10,
                  validation_data=(valid_bottleneck, to_categorical(valid_labels)))

    if verbose:
        print('Concatenate new VGG16 (without top layer) with pretrained top model.')

    vgg16_fine = build_vgg_16()
    vgg16_fine.load_weights(file_path)
    for _ in range(6):
        vgg16_fine.pop()
    vgg16_fine.add(Flatten(name='top_flatten'))    
    vgg16_fine.add(Dense(4096, activation='relu'))
    vgg16_fine.add(Dropout(0.5))
    vgg16_fine.add(Dense(4096, activation='relu'))
    vgg16_fine.add(Dropout(0.5))
    vgg16_fine.add(Dense(2, activation='softmax'))
    vgg16_fine.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    if verbose:
        print('Loading pre-trained weights into concatenated model')

    for i, layer in enumerate(reversed(top_model.layers), 1):
        pretrained_weights = layer.get_weights()
        vgg16_fine.layers[-i].set_weights(pretrained_weights)

    for layer in vgg16_fine.layers[:26]:
        layer.trainable = False

    if verbose:
        print('Layers training status:')
        for layer in vgg16_fine.layers:
            print('[%6s] %s' % ('' if layer.trainable else 'FROZEN', layer.name))        

    vgg16_fine.compile(optimizer=RMSprop(lr=1e-6), loss='binary_crossentropy', metrics=['accuracy'])

    if verbose:
        print('Train concatenated model on dogs/cats dataset sample.')

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_batches = get_batches('train', gen=train_datagen, class_mode='categorical', batch_size=BATCH_SIZE)
    valid_batches = get_batches('valid', gen=test_datagen, class_mode='categorical', batch_size=BATCH_SIZE)
    vgg16_fine.fit_generator(train_batches, epochs=100,
                             steps_per_epoch=2000 // BATCH_SIZE,
                             validation_data=valid_batches,
                             validation_steps=800 // BATCH_SIZE)
    return vgg16_fine 