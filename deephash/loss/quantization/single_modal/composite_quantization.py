from deephash.loss.quantization.base import QuantizationBase
import tensorflow as tf


class CQ(QuantizationBase):
    def __init__(self, dataset, config):
        super(CQ, self).__init__(dataset, config.output_dim, config.code_dim)
        self.config = config
        with tf.device(self.config.device):
            ### Centers shared in different modalities (image & text)
            ### Binary codes for different modalities (image & text)
            self.img_output_all = tf.placeholder(tf.float32, [None, self.config.output_dim])
            self.img_b_all = tf.placeholder(tf.float32, [None, self.config.subspace_num * self.config.subcenter_num])
            self.b_img = tf.placeholder(tf.float32, [None, self.config.subspace_num * self.config.subcenter_num])
            self.ICM_m = tf.placeholder(tf.int32, [])
            self.ICM_b_m = tf.placeholder(tf.float32, [None, self.config.subcenter_num])
            self.ICM_b_all = tf.placeholder(tf.float32, [None, self.config.subcenter_num * self.config.subspace_num])
            self.ICM_X = tf.placeholder(tf.float32, [self.config.code_batch_size, self.config.output_dim])

            self.ICM_C_m = tf.slice(self.C, [self.ICM_m * self.config.subcenter_num, 0], [self.config.subcenter_num, self.config.output_dim])
            self.ICM_X_residual = tf.add(tf.subtract(self.ICM_X, tf.matmul(self.ICM_b_all, self.C)), tf.matmul(self.ICM_b_m, self.ICM_C_m))
            ICM_X_expand = tf.expand_dims(self.ICM_X_residual, 1)
            ICM_C_m_expand = tf.expand_dims(self.ICM_C_m, 0)
            # N*sc*D  *  D*n
            word_dict = tf.constant(np.loadtxt(self.wordvec_dict), dtype=tf.float32)
            ICM_word_dict = tf.reshape(tf.matmul(tf.reshape(tf.subtract(ICM_X_expand, ICM_C_m_expand), [self.config.code_batch_size*self.config.subcenter_num, self.config.output_dim]), tf.transpose(word_dict)), [self.config.code_batch_size, self.config.subcenter_num, self.config.n_class])
            ICM_sum_squares = tf.reduce_sum(tf.square(ICM_word_dict), reduction_indices = 2)
            ICM_best_centers = tf.argmin(ICM_sum_squares, 1)
            self.ICM_best_centers_one_hot = tf.one_hot(ICM_best_centers, self.config.subcenter_num, dtype = tf.float32)
