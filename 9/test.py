def train(encoder, decoder, train_loader, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        train_loss = 0
        print('current epoch = %d' % epoch)
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28 * 28))
            #             labels = Variable(labels)

            optimizer.zero_grad()
            mu, logvar = encoder(images)
            sample = sample(mu, logvar)
            outputs = decoder(sample)

            loss = vae_loss(data, outputs, mu, logvar)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()
        print(train_loss)
