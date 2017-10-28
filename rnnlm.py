import time

import tensorflow as tf
import numpy as np


def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.

    Args:
      X: [m,n,k]
      W: [k,l]

    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


def MakeFancyRNNCell(H, keep_prob, num_layers=1):
    """Make a fancy RNN cell.

    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.

    Args:
      H: hidden state size
      keep_prob: dropout keep prob (same for input and output)
      num_layers: number of cell layers

    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    cells = []
    for _ in range(num_layers):
      cell = tf.contrib.rnn.BasicLSTMCell(H, forget_bias=0.0)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
      cells.append(cell)
    return tf.contrib.rnn.MultiRNNCell(cells)


# Decorator-foo to avoid indentation hell.
# Decorating a function as:
# @with_self_graph
# def foo(self, ...):
#     # do tensorflow stuff
#
# Makes it behave as if it were written:
# def foo(self, ...):
#     with self.graph.as_default():
#         # do tensorflow stuff
#
# We hope this will save you some indentation, and make things a bit less
# error-prone.
def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper


class RNNLM(object):
    def __init__(self, graph=None, *args, **kwargs):
        """Init function.

        This function just stores hyperparameters. You'll do all the real graph
        construction in the Build*Graph() functions below.

        Args:
          V: vocabulary size
          H: hidden state dimension
          num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, V, H, softmax_ns=200, num_layers=1):
        # Model structure; these need to be fixed for a given model.
        self.V = V
        self.H = H
        self.num_layers = num_layers

        # Training hyperparameters; these can be changed with feed_dict,
        # and you may want to do so during training.
        with tf.name_scope("Training_Parameters"):
            # Number of samples for sampled softmax.
            self.softmax_ns = softmax_ns

            self.learning_rate_ = tf.placeholder_with_default(
                0.1, [], name="learning_rate")

            # For gradient clipping, if you use it.
            # Due to a bug in TensorFlow, this needs to be an ordinary python
            # constant instead of a tf.constant.
            self.max_grad_norm_ = 5.0

            self.use_dropout_ = tf.placeholder_with_default(
                False, [], name="use_dropout")

            # If use_dropout is fed as 'True', this will have value 0.5.
            self.dropout_keep_prob_ = tf.cond(
                self.use_dropout_,
                lambda: tf.constant(0.5),
                lambda: tf.constant(1.0),
                name="dropout_keep_prob")

            # Dummy for use later.
            self.no_op_ = tf.no_op()


    @with_self_graph
    def BuildCoreGraph(self):
        """Construct the core RNNLM graph, needed for any use of the model.

        This should include:
        - Placeholders for input tensors (input_w_, initial_h_, target_y_)
        - Variables for model parameters
        - Tensors representing various intermediate states
        - A Tensor for the final state (final_h_)
        - A Tensor for the output logits (logits_), i.e. the un-normalized argument
          of the softmax(...) function in the output layer.
        - A scalar loss function (loss_)

        Your loss function should be a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).

        You shouldn't include training or sampling functions here; you'll do
        this in BuildTrainGraph and BuildSampleGraph below.

        We give you some starter definitions for input_w_ and target_y_, as
        well as a few other tensors that might help. We've also added dummy
        values for initial_h_, logits_, and loss_ - you should re-define these
        in your code as the appropriate tensors.

        See the in-line comments for more detail.
        """
        # Input ids, with dynamic shape depending on input.
        # Should be shape [batch_size, max_time] and contain integer word indices.
        self.input_w_ = tf.placeholder(tf.int32, [None, None], name="w")
        #self.input_w_ = tf.placeholder(tf.int32, [1, self.H], name="w")

        # Initial hidden state. You'll need to overwrite this with cell.zero_state
        # once you construct your RNN cell.
        self.initial_h_ = None

        # Final hidden state. You'll need to overwrite this with the output from
        # tf.nn.dynamic_rnn so that you can pass it in to the next batch (if
        # applicable).
        self.final_h_ = None

        # Output logits, which can be used by loss functions or for prediction.
        # Overwrite this with an actual Tensor of shape [batch_size, max_time]
        self.logits_ = None

        # Should be the same shape as inputs_w_
        self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")

        # Replace this with an actual loss function
        self.loss_ = None

        # Get dynamic shape info from inputs
        with tf.name_scope("batch_size"):
            self.batch_size_ = tf.shape(self.input_w_)[0]
        with tf.name_scope("max_time"):
            self.max_time_ = tf.shape(self.input_w_)[1]
        #print ("HOHO", self.batch_size_)
        #print ("HOHO2", self.input_w_, tf.shape(self.input_w_)[0])

        # Get sequence length from input_w_.
        # TL;DR: pass this to dynamic_rnn.
        # This will be a vector with elements ns[i] = len(input_w_[i])
        # You can override this in feed_dict if you want to have different-length
        # sequences in the same batch, although you shouldn't need to for this
        # assignment.
        self.ns_ = tf.tile([self.max_time_], [self.batch_size_, ], name="ns")

        #### YOUR CODE HERE ####
        # See hints in instructions!

        # Construct embedding layer
        #self.input_w_ = tf.placeholder(tf.int32, [self.batch_size_, self.max_time_])
        self.W_in_ = tf.Variable(tf.random_uniform([self.V, self.H], -1.0, 1.0), name="C")
        #self.final_h_ = 
        #C_ = tf.Variable(tf.random_uniform([V, M], -1.0, 1.0), name="C")
        self.W_in_ = tf.Variable(tf.random_uniform([self.V, self.H], -1.0, 1.0), name="C")
        self.x_ = tf.nn.embedding_lookup(self.W_in_, self.input_w_)


        # Hidden Layer
        self.cell_ = MakeFancyRNNCell(self.H, self.dropout_keep_prob_, self.num_layers)
        self.initial_h_  = self.cell_.zero_state(self.batch_size_, tf.float32)
        #self.final_h_ = tf.nn.dynamic_rnn(self.cell_, tf.expand_dims(x_,-1), initial_state=self.initial_h_)
        self.output, self.final_h_ = tf.nn.dynamic_rnn(self.cell_, self.x_, initial_state=self.initial_h_)
        #self.final_h_ = tf.reshape(self.final_h_, [-1, self.H])
        #print ("SANTA2", self.output.get_shape(), self.final_h_.c.get_shape())


        # Softmax output layer, over vocabulary. Just compute logits_ here.
        # Hint: the matmul3d function will be useful here; it's a drop-in
        # replacement for tf.matmul that will handle the "time" dimension
        # properly.
        self.W_out_ = tf.Variable(tf.random_normal([self.H, self.V]), name="W_out_")
        self.b_out_ = tf.Variable(tf.zeros([self.V,], dtype=tf.float32), name="b_out_")


        #print ("HAHA W_OUT", tf.rank(self.W_out_), self.W_out_.get_shape())
	#print ("HAHA", tf.shape(self.final_h_))
        #print ("HAHA W_OUT", tf.rank(self.final_h_), self.final_h_.get_shape())
        #self.logits_ = matmul3d(self.final_h_, self.W_out_) + self.b_out_ #Broadcasted addition
	#self.output = tf.reshape(self.output, [-1, self.H])
        #print ("HAHA", self.output.get_shape())
	self.logits_ = matmul3d(self.output, self.W_out_) + self.b_out_

        # Loss computation (true loss, for prediction)
        self.loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_y_, logits=self.logits_)
        #self.loss_ = tf.reshape(self.loss_, [-1, ])
        self.loss_ = tf.reduce_mean(self.loss_)


        #### END(YOUR CODE) ####

    @with_self_graph
    def BuildTrainGraph(self):
        """Construct the training ops.

        You should define:
        - train_loss_ : sampled softmax loss, for training
        - train_step_ : a training op that can be called once per batch

        Your loss function should be a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).
        """
        # Replace this with an actual training op
        self.train_step_ = None

        # Replace this with an actual loss function
        self.train_loss_ = None

        #### YOUR CODE HERE ####
        # See hints in instructions!

        # Define approximate loss function.
        # Note: self.softmax_ns (i.e. k=200) is already defined; use that as the
        # number of samples.
        # Loss computation (sampled, for training)

        self.x__ = tf.reshape(self.x_, [-1, self.H])
        print ("HAHA W_OUT", tf.rank(self.W_out_), self.W_out_.get_shape())
        print ("HAHA B_OUT", tf.rank(self.b_out_), self.b_out_.get_shape())
        print ("HAHA TARG Y", tf.rank(self.target_y_), self.target_y_.get_shape())
#        print ("HAHA O", tf.rank(self.x_), self.x_.get_shape())
#        print ("HAHA X__", tf.rank(self.x__), self.x__.get_shape())
        self.o = tf.reshape(self.output, [-1, self.H])
	print ("HAHA O", self.o.get_shape())
	self.t_y_ = tf.reshape(self.target_y_, [-1,1])
	print ("MERCYGO", self.t_y_.get_shape())
	print ("HOHO", self.logits_.get_shape())
	self.l_ = tf.reshape(self.logits_, [-1, self.H])
        self.per_example_train_loss_ = tf.nn.sampled_softmax_loss(weights=tf.transpose(self.W_out_), 
                biases=self.b_out_, labels=self.t_y_, inputs=self.o, num_sampled=self.softmax_ns, num_classes=self.V)
#                                             labels=tf.expand_dims(self.target_y_, 1), inputs=self.x_,
#                                             num_sampled=self.softmax_ns, num_classes=self.V)
        self.train_loss_ = tf.reduce_mean(self.per_example_train_loss_, name="sampled_softmax_loss")




        # Define optimizer and training op

        self.train_step_ = tf.train.AdagradOptimizer(self.learning_rate_).minimize(self.train_loss_)

        #### END(YOUR CODE) ####

    @with_self_graph
    def BuildSamplerGraph(self):
        """Construct the sampling ops.

        You should define pred_samples_ to be a Tensor of integer indices for
        sampled predictions for each batch element, at each timestep.

        Hint: use tf.multinomial, along with a couple of calls to tf.reshape
        """
        # Replace with a Tensor of shape [batch_size, max_time, 1]
        self.pred_samples_ = None

        #### YOUR CODE HERE ####
        # Sampled softmax loss, for training
        #pred_proba_ = tf.nn.softmax(logits_, name="pred_proba")
        #self.pred_samples_ = tf.argmax(self.logits_, 1, name="pred_max")
        self.logits__ = tf.reshape(self.logits_, [-1, self.V])
        self.pred_samples_ = tf.multinomial(self.logits__, 1, name="pred_random")
        self.pred_samples_ = tf.reshape(self.pred_samples_, [self.batch_size_, self.max_time_, 1])


        #### END(YOUR CODE) ####


