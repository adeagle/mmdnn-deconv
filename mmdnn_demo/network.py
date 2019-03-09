import tensorflow as tf
import numpy as np
np.random.seed(101)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class YOLOv3Ex(object):
    """Structure of reseau neural YOLO3"""

    # def __init__(self, x, num_classes, trainable=True, strategie_training=False, is_training=False):
    def __init__(self, x, num_classes,is_training):
        """
        Create the graph ofthe YOLOv3 model
        :param x: Placeholder for the input tensor: (normalised image (416, 416, 3)/255.)
        :param num_classes: Number of classes in the dataset
               if it isn't in the same folder as this code
        """
        self.X = x
        self.NUM_CLASSES = num_classes
        self.avg_var = []
        self.decay_bn = 0.99
        self.is_training=is_training

    def max_pool(self,_input,size,stride,name):        
        return tf.nn.max_pool(_input,[1,size,size,1],[1,stride,stride,1], padding='VALID',name=name)#208
        
    def feature_extractor(self):
        """
        Create the network graph
        :return: feature maps 5+80 in 3 grid (13,13), (26,26), (52, 52)
        """
        print("YOLOv3, let's go!!!!!!!")
        
        #self.X = tf.Print(self.X, [self.X],message='self.X: ',summarize=1000)
        
        self.conv0 = self.conv_layer(bottom=self.X, size=3, stride=1, in_channels=3, out_channels=32, name='conv_0')   #416
        
        #self.conv0 = tf.Print(self.conv0, [self.conv0],message='self.conv0: ',summarize=1000)
        
        self.maxpool1= self.max_pool(self.conv0, 2,2,name="maxpool1")#208
                  
        self.conv2 = self.conv_layer(bottom=self.maxpool1, size=3, stride=1, in_channels=32,out_channels=64, name='conv_2') 
        self.maxpool3= self.max_pool(self.conv2, 2, 2,name="maxpool3") #104 
         
        self.conv4 = self.conv_layer(bottom=self.maxpool3, size=3, stride=1, in_channels=64,out_channels=64, name='conv_4')
        self.maxpool5= self.max_pool(self.conv4, 2, 2,name="maxpool5") #52  
                      
        self.conv6 = self.conv_layer(bottom=self.maxpool5, size=3, stride=1, in_channels=64,out_channels=64, name='conv_6')
        self.maxpool7= self.max_pool(self.conv6, 2, 2,name="maxpool7") #26      
        
        #self.maxpool7 = tf.Print(self.maxpool7, [self.maxpool7],message='self.maxpool7: ',summarize=1000)                  
                      
        self.conv8 = self.conv_layer(bottom=self.maxpool7, size=3, stride=1, in_channels=64,out_channels=64, name='conv_8')
        self.maxpool9= self.max_pool(self.conv8, 2, 2, name="maxpool9") #13    
                
        self.conv10 = self.conv_layer(bottom=self.maxpool9, size=3, stride=1,in_channels=64,out_channels=128, name='conv_10')
        self.conv11 = self.conv_layer(bottom=self.conv10, size=3, stride=1, in_channels=128,out_channels=128, name='conv_11')
        self.conv12 = self.conv_layer(bottom=self.conv11, size=1, stride=1, in_channels=128,out_channels=128, name='conv_12')
        self.conv13 = self.conv_layer(bottom=self.conv12, size=3, stride=1, in_channels=128,out_channels=64, name='conv_13')
        self.conv14 = self.conv_layer(bottom=self.conv13, size=1, stride=1, in_channels=64,out_channels=18, name='conv_14',batch_norm_and_activation=False)
        
        
        self.conv15 = self.conv_layer(bottom=self.conv8, size=3, stride=1, in_channels=64,out_channels=128, name='conv_15')
        self.conv16 = self.conv_layer(bottom=self.conv15, size=3, stride=1, in_channels=128,out_channels=64, name='conv_16')
        self.conv17 = self.conv_layer(bottom=self.conv16, size=1, stride=1, in_channels=64,out_channels=18, name='conv_17',batch_norm_and_activation=False)
        
       
        self.conv18 = self.conv_layer(bottom=self.conv6, size=3, stride=1, in_channels=64,out_channels=128, name='conv_18')
        self.conv19 = self.conv_layer(bottom=self.conv18, size=3, stride=1, in_channels=128,out_channels=64, name='conv_19')
        self.conv20 = self.conv_layer(bottom=self.conv19, size=1, stride=1, in_channels=64,out_channels=18, name='conv_20',batch_norm_and_activation=False)
        
        print(self.conv14,self.conv17,self.conv20)
        
#         self.conv14 = tf.Print(self.conv14, [self.conv14],message='self.conv14: ')
#         self.conv17 = tf.Print(self.conv17, [self.conv17],message='self.conv17: ')
#         self.conv20 = tf.Print(self.conv20, [self.conv20],message='self.conv20: ')
        #v = tf.Variable(tf.truncated_normal([3, 3, 18, 18], 5., 0.01))
        
        deconv_param = tf.get_variable("deconv/param",[3, 3, 18, 18],
                                            initializer=tf.contrib.layers.xavier_initializer(),
                                            #initializer=tf.constant_initializer(-0.008087967),
                                            regularizer=tf.contrib.layers.l2_regularizer(0.0005)
                                            )
        #self.deconv_input=tf.identity(self.conv14,name="deconv_input")
        
        self.conv14 = tf.nn.conv2d_transpose(self.conv14, deconv_param, [1, 19, 19, 18], [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(
            np.random.normal(size=[18, ], loc=0.0, scale=0.01),trainable=True,dtype=np.float32, name="biases")
        
        self.conv14_add = tf.nn.relu(tf.add(self.conv14, biases))
                    
        self.deconv_output=tf.identity(self.conv14_add,name="deconv_output")
        
        return  tf.identity(self.deconv_output,name="a"),tf.identity(self.conv17,name="b"),tf.identity(self.conv20,name="c") #self.conv14,self.conv17,self.conv20
                            
    def get_var(self, initial_value, name, var_name, var_trainable=True):
        """

        :param initial_value:
        :param name:
        :param var_name:
        :param trainable:  moving average not trainable
        :return:
        """
        #return tf.Variable(initial_value, name=name+"_"+var_name, trainable=var_trainable)
        return tf.get_variable(name+"_"+var_name,shape=initial_value.shape,
                               initializer=tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(0.0005))

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal(
            [filter_size, filter_size, in_channels, out_channels], 5.0, 0.001)
        filters = self.get_var(initial_value, name, 'conv_weights')

        return filters

    def get_conv_bn_var(self, out_channels, name):
        initial_value = tf.truncated_normal([out_channels], .0, .001)
        beta = self.get_var(initial_value, name, 'bias')

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        gamma = self.get_var(initial_value, name, 'gamma')

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        mean = self.get_var(initial_value, name, 'mean', var_trainable=False)

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        variance = self.get_var(initial_value, name,
                                'variance', var_trainable=False)
        return beta, gamma, mean, variance
        
    def batch_norm(self,x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):
        with tf.variable_scope(scope):
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=decay)
    
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
    
            mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
        return normed
        
    def conv_layer(self, bottom, size, stride, in_channels, out_channels, name,batch_norm_and_activation=True):
        with tf.variable_scope(name):
            filt = self.get_conv_var(size, in_channels, out_channels, name)
            
            #filt = tf.Print(filt, [ filt],message='filt: ',summarize=1000)
            
            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')            
            
            #conv = tf.Print(conv, [ conv],message='conv: ',summarize=1000)
            
            biases = tf.Variable(
                np.random.normal(size=[out_channels, ], loc=0.0, scale=0.01),trainable=True,dtype=np.float32, name="biases")
            
            conv = tf.add(conv, biases)
                           
#             beta, gamma, moving_mean, moving_variance = self.get_conv_bn_var(out_channels, name)
#             # TODO: How to initialize moving_mean & moving_var with darknet
#             # data
#             
#             def mean_var_with_update():
#                 batch_mean, batch_var = tf.nn.moments(conv, [0, 1, 2], name='moments')
#                 train_mean = tf.assign(moving_mean,
#                                        moving_mean * self.decay_bn + batch_mean * (1 - self.decay_bn))
#                 train_var = tf.assign(moving_variance,
#                                       moving_variance * self.decay_bn + moving_variance * (1 - self.decay_bn))
#                 with tf.control_dependencies([train_mean, train_var]):
#                     return tf.identity(batch_mean), tf.identity(batch_var)
# 
#             mean, var = tf.cond(self.is_training,mean_var_with_update, lambda: (moving_mean, moving_variance))
            
            #conv = tf.Print(conv, [ conv],message=name+'conv: ',summarize=10)
            #mean = tf.Print(mean, [ mean],message=name+'mean: ',summarize=10)
            #beta = tf.Print(beta, [ beta],message=name+'beta: ',summarize=10)
            #gamma = tf.Print(gamma, [ gamma],message=name+'gamma: ',summarize=10)
                        
            if batch_norm_and_activation:
                initial_value = tf.truncated_normal([out_channels], .0, .001)
                beta = self.get_var(initial_value, name, 'bias')
        
                initial_value = tf.truncated_normal([out_channels], .0, .001)
                gamma = self.get_var(initial_value, name, 'gamma')
                        
                #normed=self.batch_norm(conv,beta,gamma,self.is_training)
                normed=tf.contrib.layers.batch_norm(conv,is_training=is_training)
                #normed = tf.nn.batch_normalization(conv, mean, var, beta, gamma, 1e-8)
                #return normed
                activation = tf.nn.relu(normed)
                return activation
            else:
                return conv

Input_shape = 608  # width=height # 608 or 416 or 320
channels = 3  # RBG
angle = 0
saturation = 1.5
exposure = 1.5
hue = 0.1
jitter = 0.3
random = 1
from tensorflow.python.framework import graph_util

g = tf.Graph()
with g.as_default():
    with tf.Session() as sess:
        X = tf.placeholder(tf.float32, shape=[None, 608, 608, 3], name='input')  # for image_data
        # Reshape images for visualization
        is_training=tf.placeholder(tf.bool, name="train_flag")
        x_reshape = tf.reshape(X, [-1, Input_shape, Input_shape, 1])
        scale1, scale2, scale3 = YOLOv3Ex(X, 1,is_training).feature_extractor()
        loss = tf.reduce_sum(scale1)+ tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) 
        #optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_local_variables())
        
        #for i in range(10):
        	#sess.run(optimizer,feed_dict={is_training:True,X:np.ones((1,608,608,3),dtype=np.float32)})
        	
        sess.run([scale1, scale2, scale3],feed_dict={is_training:True,X:np.ones((1,608,608,3))})		
        #print(scale1, scale2, scale3)
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['a','b','c'])
        
        with tf.gfile.FastGFile('model.pb', mode='wb') as f:
        	f.write(constant_graph.SerializeToString())
        
        '''
mmconvert -sf tensorflow -iw model.pb --inNodeName input  --inputShape 608,608,3 --dstNodeName a -df caffe -om tf_mobilenet
'''
