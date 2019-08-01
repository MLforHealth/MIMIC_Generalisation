"""
EmbeddingAutoencoder.py
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout, Activation, LSTM, merge, GRU, CuDNNLSTM, CuDNNGRU
from keras.objectives import categorical_crossentropy, binary_crossentropy
from keras.models import Model
from keras import backend as K
from keras import optimizers

import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

class ae_tf():
    """
    Under construction. Need a way to share graph between functions.
    """
    def __init__(self, num_input, num_hidden_1, num_hidden_2, learning_rate = 0.01):
        self.num_hidden_1=num_hidden_1
        self.num_hidden_2=num_hidden_2
        self.num_input=num_input

        self.n_epochs=5
        self.batch_size=256

        self.sess=tf.Session()

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.X=tf.placeholder("float", [None, self.num_input])
            self.Z=self.encoder(self.X)
            self.Y=self.decoder(self.Z)
            self.loss=self.get_cost()
            self.optimizer=tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            #add output to the graph
            # latent=self.encoder(self.X)

            #add input for z to graph
            self.z_=tf.placeholder("float", [None, self.num_hidden_2], name="z_")
            #dd output to the graph
            self.reconstruction=self.decoder(self.z_)


        # num_hidden_1 = 128 # 1st layer num features
        # num_hidden_2 = 32 # 2nd layer num features (the latent dim)
        # num_input = X.shape[1] # MNIST data input (img shape: 28*28)
        # num_samples=X.shape[0]

    def prepare_data(self, reconstruct=False):
        # feature=tf.placeholder(X.dtype, X.shape, name='feature')
        if reconstruct:
            self.feature=tf.placeholder("float", [None, self.num_hidden_2], name='feature')
        else:
            self.feature=tf.placeholder("float", [None, self.num_input], name='feature')
        # label=tf.placeholder(y.dtype, y.shape, name='label')

        dataset = tf.data.Dataset.from_tensor_slices((self.feature))
        if self.batch_size>0:
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.repeat().batch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        return iterator.get_next(), iterator

    def encoder(self, tensor_in):# Encoder Hidden layer with sigmoid activation #1
        with tf.device("/device:GPU:0"):
            layer_1=tf.layers.dense(inputs=tensor_in, units=self.num_hidden_1, activation=tf.nn.relu, name='layer_1')
            layer_2=tf.layers.dense(inputs=layer_1, units=self.num_hidden_2, activation=tf.nn.relu, name='layer_2')

        return layer_2

    def decoder(self, tensor_in):
        with tf.device("/device:GPU:0"):
            layer_3=tf.layers.dense(inputs=tensor_in, units=self.num_hidden_1, activation=tf.nn.relu, name='layer_3')
            layer_4=tf.layers.dense(inputs=layer_3, units=self.num_input, activation=tf.nn.relu,  name='layer_4')

        return layer_4

    def get_cost(self):
        return tf.reduce_mean(tf.pow(self.X - self.Y, 2))

    def fit(self, X, verbose=True):
        num_samples=len(X)
        next_element, iterator=self.prepare_data()
        print(tf.trainable_variables())
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2)
        
        self.sess.run(init)
        for i in range(1, self.n_epochs+1):
            self.sess.run(iterator.initializer, feed_dict={self.feature: X})
            for j in range(num_samples//self.batch_size-1):
                batch_x=self.sess.run(next_element)

                l, _= self.sess.run([self.loss, self.optimizer], feed_dict={self.X: batch_x})
                if verbose==True:
                    if j%100==1:
                        print("epoch:{}, step: {}, Loss: {}".format(i, j, l))
        save_path = self.saver.save(self.sess, "tmp/model.ckpt")
        print("Model saved in path: {}".format(save_path))

                # X_in=batch_x
                # X_out=sess.run(ae.Y, feed_dict={ae.X: batch_x})
    def transform(self, X):
        num_samples=len(X)
        self.batch_size=0
        next_element, iterator=self.prepare_data()
        all_items=[]
        self.sess.run(iterator.initializer, feed_dict={self.feature: X})
        # for j in range(num_samples):
        for s in range(num_samples):
            batch_x=self.sess.run(next_element)
            batch_x=batch_x.reshape(1, len(batch_x))
            all_items.append(np.squeeze(self.sess.run(self.Z, feed_dict={self.X: batch_x})))
        return np.asarray(all_items)

    def predict(self, X):
        #add output to the graph

        num_samples=len(X)
        self.batch_size=0
        next_element, iterator=self.prepare_data()
        all_items=[]

        self.sess.run(iterator.initializer, feed_dict={self.feature: X})
        # for j in range(num_samples):
        for s in range(num_samples):
            batch_x=self.sess.run(next_element)
            batch_x=batch_x.reshape(1, len(batch_x))
            all_items.append(np.squeeze(self.sess.run(self.Y, feed_dict={self.X: batch_x})))
        return np.asarray(all_items)

    def reconstruct(self, Z_in):
        num_samples=len(Z_in)
        self.batch_size=0
        next_element, iterator=self.prepare_data(reconstruct=True)
        all_items=[]
        self.sess.run(iterator.initializer, feed_dict={self.feature: Z_in})
        # for j in range(num_samples):
        for s in range(num_samples):
            batch_z=self.sess.run(next_element)
            batch_z=batch_z.reshape(1, len(batch_z))
            all_items.append(np.squeeze(self.sess.run(self.reconstruction, feed_dict={self.z_: batch_z})))
        return np.asarray(all_items)

    def restore(self, path=""):
        if len(path)==0:
            path="tmp/model.ckpt"
        #load model
        latest_checkpoint = tf.train.latest_checkpoint("tmp/")
        self.saver = tf.train.import_meta_graph(latest_checkpoint+".meta")
        self.saver.restore(self.sess, latest_checkpoint)





class ae_keras():
    def __init__(self, num_input, num_hidden_1, num_hidden_2, lr=0.001):
        self.input_size=num_input
        self.hidden_size=num_hidden_1
        self.code_size=num_hidden_2
        self.lr=lr
        self.build_model()


        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
        config.gpu_options.allow_growth = True  # Don't use all GPUs
        config.allow_soft_placement = True  # Enable manual control
        sess = tf.Session(config=config)
        K.set_session(sess)

    def get_params(self):
        return self.input_size, self.hidden_size, self.code_size, self.lr



    def encoder(self, input_img):
        #encoder
        # input_img = Input(shape=(input_size,))
        with tf.device("/device:GPU:2"):
            hidden_1 = Dense(self.hidden_size, activation='relu', name='l1')(input_img)
            latent = Dense(self.code_size, activation='relu', name='l2')(hidden_1)
        return latent

    def decoder(self, latent):
        with tf.device("/device:GPU:2"):
            hidden_2 = Dense(self.hidden_size, activation='relu', name='l3')(latent)
            output_img = Dense(self.input_size, activation='sigmoid', name='l4')(hidden_2)
        return output_img

    def build_model(self):
        input_img = Input(shape=(self.input_size,))
        self.model=Model(input_img, self.decoder(self.encoder(input_img)))
        # self.model.compile(optimizer='adam', loss='binary_crossentropy') # only for positive values between 0 and 1
        opt=optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer=opt, loss='mean_squared_error')
        self.latent_model=Model(input_img, self.encoder(input_img))

    def load_model(self):
        # load json and create model
        json_file = open('tmp/ae_keras.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("tmp/ae_keras.h5")
        print("Loaded model from disk")
         
        # evaluate loaded model on test data
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


    def fit(self, x_train, epochs=2, save=False):
        """
        fit the model
        """
        history=self.model.fit(x_train, x_train, epochs=epochs, batch_size=512,shuffle=True)
        loss=history.history['loss']

        if save==True:
            # serialize model to JSON
            model_json = self.model.to_json()
            with open("tmp/ae_keras.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights("tmp/ae_keras.h5")
            print("Saved model to disk")

        return loss

    def transform(self, X):
        """
        transform to layer latent representation
        """
        input_img = Input(shape=(self.input_size,))
        hidden_1 = Dense(self.hidden_size, activation='relu', weights=self.model.get_layer("l1").get_weights())(input_img)
        latent = Dense(self.code_size, activation='relu',  weights=self.model.get_layer("l2").get_weights())(hidden_1)
        self.latent_model=Model(input_img, latent)
        return self.latent_model.predict(X)

    def predict(self, X):
        """
        transform to self.code layer latent representation
        """
        return self.model.predict(X)

    def reconstruct(self, Z):
        """
        transform to layer latent representation
        """
        latent = Input(shape=(self.code_size,))
        hidden_2 = Dense(self.hidden_size, activation='relu', weights=self.model.get_layer("l3").get_weights())(latent)
        output_img = Dense(self.input_size, activation='sigmoid', weights=self.model.get_layer("l4").get_weights())(hidden_2)
        self.reconstruction_model=Model(latent, output_img)
        return self.reconstruction_model.predict(Z)




class rnn_tf():
    """
    Under construction. Need a way to share graph between functions.
    """
    def __init__(self, input_shape=(None, None), hidden_layer_size=128, activation='tanh', modeltype='lstm', n_epochs=5, learning_rate = 0.01, input_dropout=0.0, output_dropout=0.3, optimizer='adam'):
        self.hidden_layer_size=hidden_layer_size
        self.length, self.n_features=input_shape
        self.activation=activation
        self.reg=0.2
        self.num_classes=2
        self.class_weights=[1,9]

        self.input_dropout=input_dropout
        self.output_dropout=output_dropout
        self.learning_rate=learning_rate

        self.n_epochs=n_epochs


        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))
        config.gpu_options.allow_growth = True  # Don't use all GPUs
        config.allow_soft_placement = True  # Enable manual control

        self.sess=tf.Session(config = config)



        #placeholders
        self.feature=tf.placeholder("float", [None, self.length, self.n_features], name='feature')
        self.label = tf.placeholder('float', [None], name='label')
        

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            # self.data=tf.placeholder(tf.float32, [time_steps, batch_size, num_features])
            # self.model= tf.contrib.rnn.BasicLSTMCell(lstm_size)
            # self.state = lstm.zero_state(batch_size, dtype=tf.float32)

            # self.xplaceholder= tf.placeholder('float',[None,self.n_features], name='xplaceholder')
            # self.yplaceholder = tf.placeholder('float', [None], name='yplaceholder')
            # with tf.device("/device:GPU:0"):
            self.build_cell(modeltype)
            self.logits, self.predictions=self.do_prediction()
            self.cls_prediction=tf.argmax(self.logits, axis=1, name='cls_prediction')

        self.loss=self.get_cost()
        print(self.predictions)
        print(self.cls_prediction)
        self.accuracy=tf.metrics.accuracy(self.label,self.predictions[:,1])
        

        #use different optimizers
        if optimizer=='rmsprop':
            self.optimizer=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        elif optimizer=='adagrad':
            self.optimizer=tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
        else:
            self.optimizer=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        

        with tf.name_scope('performance'):
            tf_loss_ph = tf.placeholder(tf.float32,shape=None,name='loss_summary')
            tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)
            tf_accuracy_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
            tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph)
        performance_summaries = tf.summary.merge([tf_loss_summary,tf_accuracy_summary])
            

    def build_cell(self, modeltype):
        """
        build lstm or gru cell
        """
        if modeltype=='lstm':
            rnn_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_layer_size, activation=self.activation, reuse=tf.AUTO_REUSE, initializer=tf.glorot_uniform_initializer())
            rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_layer_size, activation=self.activation, reuse=tf.AUTO_REUSE, initializer=tf.glorot_uniform_initializer())
            # rnn_cell=tf.keras.layers.LSTMCell(self.hidden_layer_size, activation=self.activation)
        elif modeltype== 'gru':
            rnn_cell = tf.nn.rnn_cell.GRUCell(self.hidden_layer_size, activation=self.activation, reuse=tf.AUTO_REUSE)
            rnn_cell_bw = tf.nn.rnn_cell.GRUCell(self.hidden_layer_size, activation=self.activation, reuse=tf.AUTO_REUSE)
        else:
            raise('Unknown modeltype for RNN')

        self.rnn_cell=tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=1.0-self.input_dropout, output_keep_prob=1.0-self.output_dropout)
        self.rnn_cell_bw=tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bw, input_keep_prob=1.0-self.input_dropout, output_keep_prob=1.0-self.output_dropout)
        # self.rnn_cell=rnn_cell

    def do_prediction(self):
        """
        """
        layer ={ 'weights': tf.truncated_normal([self.hidden_layer_size*2, self.num_classes], stddev=0.001),'bias': tf.Variable(tf.random_normal([self.num_classes]))}
        # weight = tf.truncated_normal([self.hidden_layer_size, self.num_classes], stddev=0.01)
        # bias = tf.constant(0.1, shape=[self.num_classes])

        # Recurrent network.
        # NOTE: API claims these are equivalent, but they are not
        # try:
        output, final_state=tf.nn.dynamic_rnn(self.rnn_cell, self.feature, dtype=tf.float32)
        print(output)

        self.output, self.final_state=tf.nn.bidirectional_dynamic_rnn(self.rnn_cell, self.rnn_cell_bw, self.feature, dtype=tf.float32)
        #output of bidirectional lstm is a tuple so concatenate the two outputs:
        self.output = tf.concat(self.output, 2)
        # except:
        #     #this is not working like the API says it would. defaulting to the depricated version
        #     self.output, self.final_state=tf.keras.layers.RNN(self.rnn_cell, self.feature, dtype=tf.float32)
        # self.output, self.final_state=tf.keras.layers.Bidirectional(tf.keras.layers.RNN(self.rnn_cell, self.feature, dtype=tf.float32, return_sequences=True))

        print(self.output)

        with tf.variable_scope("model") as scope:
            tf.get_variable_scope().reuse_variables()

            # flatten + sigmoid
            # logits = tf.matmul(self.final_state[-1], layer['weights']) + layer['bias']
            logits = tf.matmul(self.output[:, -1, :], layer['weights']) + layer['bias']
            prediction = tf.nn.softmax(logits) #logits?
            
        return logits, prediction


    def prepare_data(self, batch_size, label=None, shuffle=True):
        """
        """
        if label is None:
            dataset=tf.data.Dataset.from_tensor_slices((self.feature))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.feature, self.label))
        
        if batch_size>0:
            if shuffle:
                dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.repeat().batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        return iterator.get_next(), iterator


    # def get_cost(self):
    #     predictions = self.logits
    #     real = tf.cast(tf.squeeze(self.label), tf.int32)

    #     class_weight = tf.expand_dims(tf.cast(self.class_weights, tf.int32), axis=0)
    #     # print("class_weights", class_weight)
    #     one_hot_labels = tf.cast(tf.one_hot(real, depth=self.num_classes), tf.int32)
    #     weight_per_label = tf.cast(tf.transpose(tf.matmul(one_hot_labels, tf.transpose(class_weight))), tf.float32) #shape [1, batch_size]

    #     xent = tf.multiply(weight_per_label, tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=predictions, name="xent_raw")) #shape [1, batch_size]
    #     loss = tf.reduce_mean(xent) #shape 1
    #     ce = loss
    #     l2 = self.reg * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    #     ce += l2
    #     return ce

    # def get_cost(self):
    #     # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.label))
    #     return tf.losses.mean_squared_error(self.label, self.predictions)

    # def get_cost(self):
    #     return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(tf.squeeze(self.label), tf.int32)))

    def get_cost(self):
        # return tf.reduce_mean(categorical_crossentropy(self.label, self.predictions))
        return tf.reduce_mean(binary_crossentropy(self.label, self.predictions[:,1]))

    def fit(self, X, y, epochs=5, validation_data=None, verbose=True, batch_size=1, save=False):
        
        num_samples=len(X)
        next_element, iterator=self.prepare_data(batch_size, label=self.label)

        if not os.path.exists(os.path.join(os.getcwd(),'summaries/')):
            os.mkdir('summaries')
        if not os.path.exists(os.path.join('summaries','first')):
            os.mkdir(os.path.join('summaries','first'))

        summ_writer = tf.summary.FileWriter(os.path.join('summaries','first'), self.sess)

        # print(tf.trainable_variables())
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2)
        
        self.sess.run(init)
        train_loss=0
        for i in range(1, epochs+1):
            self.sess.run(iterator.initializer, feed_dict={self.feature: X, self.label:y})

            for j in range(num_samples//batch_size-1):
                batch_x, batch_y=self.sess.run(next_element)

                l, a,  _= self.sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={self.feature: batch_x, self.label:batch_y})
                train_loss+=l
                predictions=self.sess.run(self.cls_prediction, feed_dict={self.feature:batch_x})
                print(predictions.shape)
                print(np.sum(predictions))
                print(np.sum(batch_y))
                if verbose==True:
                    if j%100==1:
                        print("epoch:{}, step: {}, Loss: {}".format(i, j, l))
                summ = session.run(performance_summaries, feed_dict={tf_loss_ph:l, tf_accuracy_ph:a})
                summ_writer.add_summary(summ, epoch)

        if save==True:
            save_path = self.saver.save(self.sess, "tmp/lstm_model.ckpt")
            print("Model saved in path: {}".format(save_path))

        # validation
        if validation_data is None:
            validation_data=(np.zeros_like(X[:3]), np.zeros_like(y[:3]))
        X_val, y_val=validation_data
        num_samples=len(X_val)
        self.sess.run(iterator.initializer, feed_dict={self.feature: X_val, self.label:y_val})
        loss=0
        for j in range(num_samples//batch_size-1):
            batch_x, batch_y=self.sess.run(next_element)
            l= self.sess.run(self.loss, self.accuracy, feed_dict={self.feature: batch_x, self.label:batch_y})
            loss+=l


        
        return loss



    def predict(self, X):
        #add output to the graph

        num_samples=len(X)
        batch_size=1
        next_element, iterator=self.prepare_data(batch_size, shuffle=False)
        all_items=[]

        self.sess.run(iterator.initializer, feed_dict={self.feature: X})
        for s in range(num_samples):
            batch_x=self.sess.run(next_element)
            # batch_x=np.expand_dims(batch_x, axis=0)
            # batch_x=batch_x.reshape(1, len(batch_x))
            all_items.append(np.squeeze(self.sess.run(self.predictions, feed_dict={self.feature: batch_x})))
        # print(batch_x.shape)
        # print(np.asarray(all_items).shape)
        return np.asarray(all_items)[:,1]


    def restore(self, path=""):
        if len(path)==0:
            path="tmp/lstm_model.ckpt"
        #load model
        latest_checkpoint = tf.train.latest_checkpoint("tmp/")
        self.saver = tf.train.import_meta_graph(latest_checkpoint+".meta")
        self.saver.restore(self.sess, latest_checkpoint)











class rnn_tf2():
    """
    Under construction. Need a way to share graph between functions.
    """
    def __init__(self, input_shape=(None, None), hidden_layer_size=128, activation='tanh', modeltype='lstm', n_epochs=5, learning_rate = 0.01, input_dropout=0.0, output_dropout=0.3, optimizer='adam'):
        self.hidden_layer_size=hidden_layer_size
        self.length, self.n_features=input_shape
        self.activation=activation
        self.reg=0.2
        self.num_classes=2
        self.class_weights=[1,9]

        self.input_dropout=input_dropout
        self.output_dropout=output_dropout
        self.learning_rate=learning_rate

        self.n_epochs=n_epochs


        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))
        config.gpu_options.allow_growth = True  # Don't use all GPUs
        config.allow_soft_placement = True  # Enable manual control

        self.sess=tf.Session(config = config)



        #placeholders
        self.feature=tf.placeholder("float", [None, self.length, self.n_features], name='feature')
        self.label = tf.placeholder('float', [None], name='label')
        

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            # self.data=tf.placeholder(tf.float32, [time_steps, batch_size, num_features])
            # self.model= tf.contrib.rnn.BasicLSTMCell(lstm_size)
            # self.state = lstm.zero_state(batch_size, dtype=tf.float32)

            # self.xplaceholder= tf.placeholder('float',[None,self.n_features], name='xplaceholder')
            # self.yplaceholder = tf.placeholder('float', [None], name='yplaceholder')
            # with tf.device("/device:GPU:0"):
            self.build_cell(modeltype)
            # self.logits, self.predictions=self.do_prediction()
            # self.cls_prediction=tf.reduce_max(self.predictions, axis=1)
            self.predictions=self.do_prediction()

        self.loss=self.get_cost()        
        self.accuracy=tf.metrics.accuracy(self.label,tf.greater_equal(tf.squeeze(self.predictions), 0.5))
        #use different optimizers
        if optimizer=='rmsprop':
            self.optimizer=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        elif optimizer=='adagrad':
            self.optimizer=tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
        else:
            self.optimizer=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        with tf.name_scope('performance'):
            self.tf_loss_ph = tf.placeholder(tf.float32,shape=None,name='loss_summary')
            self.tf_loss_summary = tf.summary.scalar('loss', self.tf_loss_ph)
            self.tf_val_loss_summary= tf.summary.scalar('val loss', self.tf_loss_ph)
            self.tf_accuracy_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
            self.tf_accuracy_summary = tf.summary.scalar('accuracy', self.tf_accuracy_ph)
        self.performance_summaries = tf.summary.merge([self.tf_loss_summary,self.tf_accuracy_summary])
            

    def build_cell(self, modeltype):
        """
        build lstm or gru cell
        """
        if modeltype=='lstm':
            # kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', 
            # not zeros
            # rnn_out_f = LSTM(self.hidden_layer_size, input_shape=(self.length ,self.n_features), use_bias=True, activation=self.activation, dropout=self.output_dropout, recurrent_dropout=self.input_dropout,                  )(self.feature)
            # rnn_out_b = LSTM(self.hidden_layer_size, input_shape=(self.length ,self.n_features), use_bias=True, activation=self.activation, dropout=self.output_dropout, recurrent_dropout=self.input_dropout, go_backwards=True)(self.feature)

            #zeros failing spontaneously.
            rnn_out_f = CuDNNLSTM(self.hidden_layer_size, input_shape=(self.length ,self.n_features),                  )(self.feature)
            rnn_out_b = CuDNNLSTM(self.hidden_layer_size, input_shape=(self.length ,self.n_features), go_backwards=True)(self.feature)

            #zeros
            # rnn_cell=tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.hidden_layer_size)
            # rnn_cell_bw=tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.hidden_layer_size)

            #zeros
            # rnn_cell=tf.contrib.cudnn_rnn.CudnnLSTM(1, self.hidden_layer_size, dropout=self.output_dropout, kernel_initializer=tf.glorot_uniform_initializer(), bias_initializer='zeros', direction='bidirectional')
            # rnn_cell_bw=tf.contrib.cudnn_rnn.CudnnLSTM(1, self.hidden_layer_size, dropout=self.output_dropout, kernel_initializer=tf.glorot_uniform_initializer(), bias_initializer='zeros', direction='bidirectional')



            # rnn_cell=tf.keras.layers.LSTMCell(self.hidden_layer_size, activation=self.activation)
        elif modeltype== 'gru':
            rnn_out_f = CuDNNGRU(self.hidden_layer_size, input_shape=(self.length ,self.n_features),                  )(self.feature)
            rnn_out_b = CuDNNGRU(self.hidden_layer_size, input_shape=(self.length ,self.n_features), go_backwards=True)(self.feature)
        else:
            raise('Unknown modeltype for RNN')

        # self.rnn_cell=tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=1.0-self.input_dropout, output_keep_prob=1.0-self.output_dropout)
        # self.rnn_cell_bw=tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bw, input_keep_prob=1.0-self.input_dropout, output_keep_prob=1.0-self.output_dropout)

        self.rnn_cell = merge.Concatenate(axis=-1)([rnn_out_f, rnn_out_b])
        self.rnn_cell = Dropout(self.output_dropout) (self.rnn_cell)

    # def do_prediction(self):
    #     """
    #     """
    #     layer ={ 'weights': tf.truncated_normal([self.hidden_layer_size*2, self.num_classes], stddev=0.001),'bias': tf.Variable(tf.random_normal([self.num_classes]))}
    #     # weight = tf.truncated_normal([self.hidden_layer_size, self.num_classes], stddev=0.01)
    #     # bias = tf.constant(0.1, shape=[self.num_classes])

    #     # Recurrent network.
    #     # NOTE: API claims these are equivalent, but they are not
    #     # try:
    #     # rnn_output = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell, self.rnn_cell_bw, self.feature, dtype=tf.float32,time_major=True)
    #     output, final_state=tf.nn.dynamic_rnn(self.rnn_cell, self.feature, dtype=tf.float32)
    #     print(output)

    #     self.output, self.final_state=tf.nn.bidirectional_dynamic_rnn(self.rnn_cell, self.rnn_cell_bw, self.feature, dtype=tf.float32)
    #     #output of bidirectional lstm is a tuple so concatenate the two outputs:
    #     self.output = tf.concat(self.output, 2)
    #     # except:
    #     #     #this is not working like the API says it would. defaulting to the depricated version
    #     #     self.output, self.final_state=tf.keras.layers.RNN(self.rnn_cell, self.feature, dtype=tf.float32)
    #     # self.output, self.final_state=tf.keras.layers.Bidirectional(tf.keras.layers.RNN(self.rnn_cell, self.feature, dtype=tf.float32, return_sequences=True))

    #     print(self.output)

    #     with tf.variable_scope("model") as scope:
    #         tf.get_variable_scope().reuse_variables()

    #         # flatten + sigmoid
    #         # logits = tf.matmul(self.final_state[-1], layer['weights']) + layer['bias']
    #         logits = tf.matmul(self.output[:, -1, :], layer['weights']) + layer['bias']
    #         prediction = tf.nn.softmax(logits) #logits?
            
    #     return logits, prediction

    def do_prediction(self):
        """
        """
        pred = Dense(1, activation='sigmoid')(self.rnn_cell)
        # prediction=Activation('sigmoid')(logits)

        return pred


    def prepare_data(self, batch_size, label=None, shuffle=True):
        """
        """
        if label is None:
            dataset=tf.data.Dataset.from_tensor_slices((self.feature))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.feature, self.label))
        
        if batch_size>0:
            if shuffle:
                dataset = dataset.shuffle(buffer_size=100)
            dataset = dataset.repeat().batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        return iterator.get_next(), iterator


    def get_cost(self):
        # return tf.reduce_mean(categorical_crossentropy(self.label, self.predictions))
        return tf.reduce_mean(binary_crossentropy(tf.squeeze(self.label), tf.squeeze(self.predictions)))

    def fit(self, X, y, epochs=5, validation_data=None, verbose=True, batch_size=1, save=False):
        
        num_samples=len(X)
        next_element, iterator=self.prepare_data(batch_size, label=self.label)

        if not os.path.exists(os.path.join(os.getcwd(),'summaries/')):
            os.mkdir('summaries')
        if not os.path.exists(os.path.join('summaries','first')):
            os.mkdir(os.path.join('summaries','first'))

        summ_writer = tf.summary.FileWriter(os.path.join('summaries','first'), self.sess.graph)


        # print(tf.trainable_variables())
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2)
        print("num_samples {}, batch {}, iter {}".format(num_samples, batch_size, num_samples//batch_size-1))
        
        self.sess.run(init)
        for i in range(1, epochs+1):
            self.sess.run(iterator.initializer, feed_dict={self.feature: X, self.label:y})
            for j in range(num_samples//batch_size-1):
                batch_x, batch_y=self.sess.run(next_element)

                l, _= self.sess.run([self.loss, self.optimizer], feed_dict={self.feature: batch_x, self.label:batch_y, K.learning_phase():1})
                # predictions=self.sess.run(self.cls_prediction, feed_dict={self.feature:batch_x})
                if verbose==True:
                    if j%100==1:
                        print("epoch:{}, step: {}, Loss: {}".format(i, j, l))
                # print(l, a)
                summ = self.sess.run(self.tf_loss_summary, feed_dict={self.tf_loss_ph:l})
                summ_writer.add_summary(summ, j+i*(num_samples//batch_size-1))

        if save==True:
            save_path = self.saver.save(self.sess, "tmp/lstm_model.ckpt")
            print("Model saved in path: {}".format(save_path))

        # validation
        if validation_data is None:
            return 0
        X_val, y_val=validation_data
        num_samples=len(X_val)
        batch_size=2
        next_element, iterator=self.prepare_data(batch_size, label=self.label, shuffle=False)
        self.sess.run(iterator.initializer, feed_dict={self.feature: X_val, self.label:y_val})
        loss=0
        # print(num_samples)
        for j in range(num_samples//batch_size):
            # print(j)
            batch_x, batch_y=self.sess.run(next_element)

            # print(batch_x.shape, batch_y.shape)
            l= self.sess.run(self.loss, feed_dict={self.feature: batch_x, self.label:batch_y})
            loss+=l

        summ = self.sess.run(self.tf_val_loss_summary, feed_dict={self.tf_loss_ph:loss})
        summ_writer.add_summary(summ, j)
        return loss



    def predict(self, X):
        #add output to the graph

        num_samples=len(X)
        batch_size=1
        next_element, iterator=self.prepare_data(batch_size, shuffle=False)
        all_items=[]

        self.sess.run(iterator.initializer, feed_dict={self.feature: X})
        for s in range(num_samples):
            batch_x=self.sess.run(next_element)
            # batch_x=np.expand_dims(batch_x, axis=0)
            # batch_x=batch_x.reshape(1, len(batch_x))
            all_items.append(np.squeeze(self.sess.run(self.predictions, feed_dict={self.feature: batch_x})))
        # print(batch_x.shape)
        # print(np.asarray(all_items).shape)
        return np.asarray(all_items)


    def restore(self, path=""):
        if len(path)==0:
            path="tmp/lstm_model.ckpt"
        #load model
        latest_checkpoint = tf.train.latest_checkpoint("tmp/")
        self.saver = tf.train.import_meta_graph(latest_checkpoint+".meta")
        self.saver.restore(self.sess, latest_checkpoint)

    def reset(self):
        self.sess.close()
        tf.reset_default_graph()
        self.sess=tf.Session()



def main(model='tf'):
    # load the data
    (x_train, _), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    num_hidden_1 = 128 # 1st layer num features
    num_hidden_2 = 32 # 2nd layer num features (the latent dim)
    num_input = x_train.shape[1] # MNIST data input (img shape: 28*28)

    # build a model
    if model=='tf':
        ae=ae_tf(num_input, num_hidden_1, num_hidden_2)
    else:
        ae=ae_keras(num_input, num_hidden_1, num_hidden_2)

    # fit the data
    ae.fit(x_train)

    #view the reconstruction
    decoded_imgs=ae.predict(x_test)


    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    #view the transformde images
    # print(x_test.shape)
    latent_imgs=ae.transform(x_test)
    sorter = np.argsort(y_test)
    # print(latent_imgs.shape)
    # print(np.repeat(latent_imgs[sorter], 50, axis=1).shape)
    
    # print(y_test[sorter])
    plt.imshow(np.repeat(latent_imgs[sorter], 50, axis=1))
    plt.show()

    #transform these back into images
    decoded_imgs=ae.reconstruct(latent_imgs)
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__=="__main__":
    # main(model='keras')
    # main(model='keras')
    main(model='tf')
