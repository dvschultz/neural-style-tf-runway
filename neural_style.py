import tensorflow as tf
import numpy as np 
import scipy.io  
import argparse 
import struct
import errno
import time                       
import cv2
import os
from PIL import Image

class NeuralStyle():

  def __init__(self):

    self.verbose = True
    self.device_opts = '/gpu:0' #'/cpu:0'
    self.print_iterations = 50
    self.style_imgs_weights = [1.0]
    self.model_weights = 'imagenet-vgg-verydeep-19.mat'
    self.init_img_type = 'content' #['random', 'content', 'style']
    self.content_weight = 5e0
    self.style_weight = 1e4
    self.learning_rate = 1e0
    self.optimizer_type = 'lbfgs' #['lbfgs', 'adam']
    self.max_size = 360
    self.max_iterations = 200
    self.style_scale = 0.5
    self.tv_weight = 1e-3
    self.temporal_weight = 2e2
    self.content_loss_function = 1 #[1, 2, 3]
    self.content_layers = ['conv4_2']
    self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    self.pooling_type = 'avg' #['avg', 'max']
    self.seed = 10 #options['seed']
    self.noise_ratio = 1.0
    self.original_colors = False
    self.content_layer_weights = [1.0]
    self.style_layer_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    # original_colors = False
    self.color_convert_type = 'yuv' #['yuv', 'ycrcb', 'luv', 'lab']
    self.color_convert_time = 'after' #['after', 'before']
    # video = False

    self.style_layer_weights = self.normalize(self.style_layer_weights)
    self.content_layer_weights = self.normalize(self.content_layer_weights)
    self.style_imgs_weights = self.normalize(self.style_imgs_weights)
    

  def run(self,content,style1,og_colors,max_iter,maxsize,scale):
    #print('maxsize: %05d' %  maxsize)
    #print('max_size: %05d' % self.max_size)
    self.original_colors = og_colors
    self.max_iterations = max_iter
    self.max_size = maxsize
    print('max_size: %05d' % self.max_size)
    self.style_scale = scale
    content_img = self.get_content_image(content)
    style_imgs = self.get_style_images(content_img,style1)
    stylized_img = self.render_single_image(content_img,style_imgs)
    finished_img = self.convert_to_pil(stylized_img)
    return finished_img

  def get_content_image(self,content_img):
    print('MAX SIZE:%05d' % self.max_size)
    # https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
    pil_image = content_img.convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    img = open_cv_image.astype(np.float32)
    h, w, d = img.shape
    mx = self.max_size
    # resize if > max size
    if h > w and h > mx:
      w = (float(mx) / float(h)) * w
      img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
      h = (float(mx) / float(w)) * h
      img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
    img = self.preprocess(img)
    return img

  def get_style_images(self,content_img,style_img):
    _, ch, cw, cd = content_img.shape

    pil_image = style_img.convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    mx = self.max_size
    style_imgs = []
    # for style_fn in args.style_imgs:
    # path = os.path.join(args.style_imgs_dir, style_fn)
    # bgr image
    # img = cv2.imread(path, cv2.IMREAD_COLOR)
    # check_image(img, path)
    img = open_cv_image.astype(np.float32)
    sh, sw, sd = img.shape

    # use scale args to resize and tile image
    scaled_img = cv2.resize(img, dsize=(int(sw*self.style_scale), int(sh*self.style_scale)), interpolation=cv2.INTER_AREA)
    ssh, ssw, ssd = scaled_img.shape
    
    if ssh > ch and ssw > cw:
      starty = int((ssh-ch)/2)
      startx = int((ssw-cw)/2)
      img = scaled_img[starty:starty+ch, startx:startx+cw]
    elif ssh > ch:
      starty = int((ssh-ch)/2)
      img = scaled_img[starty:starty+ch, 0:ssw]
      if ssw != cw:
        img = cv2.copyMakeBorder(img,0,0,0,(cw-ssw),cv2.BORDER_REFLECT)
    elif ssw > cw:
      startx = int((ssw-cw)/2)
      img = scaled_img[0:ssh, startx:startx+cw]
      if ssh != ch:
        img = cv2.copyMakeBorder(img,0,(ch-ssh),0,0,cv2.BORDER_REFLECT)
    else:
      img = cv2.copyMakeBorder(scaled_img,0,(ch-ssh),0,(cw-ssw),cv2.BORDER_REFLECT)

    img = self.preprocess(img)
    style_imgs.append(img)
    return style_imgs
    # return img

  def convert_to_pil(self,img):
    post = self.postprocess(img)
    return Image.fromarray(post)


  def build_model(self,input_img):
    if self.verbose: print('\nBUILDING VGG-19 NETWORK')
    net = {}
    _, h, w, d     = input_img.shape
    
    if self.verbose: print('loading model weights...')
    vgg_rawnet     = scipy.io.loadmat(self.model_weights)
    vgg_layers     = vgg_rawnet['layers'][0]
    if self.verbose: print('constructing layers...')
    net['input']   = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))

    if self.verbose: print('LAYER GROUP 1')
    net['conv1_1'] = self.conv_layer('conv1_1', net['input'], W=self.get_weights(vgg_layers, 0))
    net['relu1_1'] = self.relu_layer('relu1_1', net['conv1_1'], b=self.get_bias(vgg_layers, 0))

    net['conv1_2'] = self.conv_layer('conv1_2', net['relu1_1'], W=self.get_weights(vgg_layers, 2))
    net['relu1_2'] = self.relu_layer('relu1_2', net['conv1_2'], b=self.get_bias(vgg_layers, 2))
    
    net['pool1']   = self.pool_layer('pool1', net['relu1_2'])

    if self.verbose: print('LAYER GROUP 2')  
    net['conv2_1'] = self.conv_layer('conv2_1', net['pool1'], W=self.get_weights(vgg_layers, 5))
    net['relu2_1'] = self.relu_layer('relu2_1', net['conv2_1'], b=self.get_bias(vgg_layers, 5))
    
    net['conv2_2'] = self.conv_layer('conv2_2', net['relu2_1'], W=self.get_weights(vgg_layers, 7))
    net['relu2_2'] = self.relu_layer('relu2_2', net['conv2_2'], b=self.get_bias(vgg_layers, 7))
    
    net['pool2']   = self.pool_layer('pool2', net['relu2_2'])
    
    if self.verbose: print('LAYER GROUP 3')
    net['conv3_1'] = self.conv_layer('conv3_1', net['pool2'], W=self.get_weights(vgg_layers, 10))
    net['relu3_1'] = self.relu_layer('relu3_1', net['conv3_1'], b=self.get_bias(vgg_layers, 10))

    net['conv3_2'] = self.conv_layer('conv3_2', net['relu3_1'], W=self.get_weights(vgg_layers, 12))
    net['relu3_2'] = self.relu_layer('relu3_2', net['conv3_2'], b=self.get_bias(vgg_layers, 12))

    net['conv3_3'] = self.conv_layer('conv3_3', net['relu3_2'], W=self.get_weights(vgg_layers, 14))
    net['relu3_3'] = self.relu_layer('relu3_3', net['conv3_3'], b=self.get_bias(vgg_layers, 14))

    net['conv3_4'] = self.conv_layer('conv3_4', net['relu3_3'], W=self.get_weights(vgg_layers, 16))
    net['relu3_4'] = self.relu_layer('relu3_4', net['conv3_4'], b=self.get_bias(vgg_layers, 16))

    net['pool3']   = self.pool_layer('pool3', net['relu3_4'])

    if self.verbose: print('LAYER GROUP 4')
    net['conv4_1'] = self.conv_layer('conv4_1', net['pool3'], W=self.get_weights(vgg_layers, 19))
    net['relu4_1'] = self.relu_layer('relu4_1', net['conv4_1'], b=self.get_bias(vgg_layers, 19))

    net['conv4_2'] = self.conv_layer('conv4_2', net['relu4_1'], W=self.get_weights(vgg_layers, 21))
    net['relu4_2'] = self.relu_layer('relu4_2', net['conv4_2'], b=self.get_bias(vgg_layers, 21))

    net['conv4_3'] = self.conv_layer('conv4_3', net['relu4_2'], W=self.get_weights(vgg_layers, 23))
    net['relu4_3'] = self.relu_layer('relu4_3', net['conv4_3'], b=self.get_bias(vgg_layers, 23))

    net['conv4_4'] = self.conv_layer('conv4_4', net['relu4_3'], W=self.get_weights(vgg_layers, 25))
    net['relu4_4'] = self.relu_layer('relu4_4', net['conv4_4'], b=self.get_bias(vgg_layers, 25))

    net['pool4']   = self.pool_layer('pool4', net['relu4_4'])

    if self.verbose: print('LAYER GROUP 5')
    net['conv5_1'] = self.conv_layer('conv5_1', net['pool4'], W=self.get_weights(vgg_layers, 28))
    net['relu5_1'] = self.relu_layer('relu5_1', net['conv5_1'], b=self.get_bias(vgg_layers, 28))

    net['conv5_2'] = self.conv_layer('conv5_2', net['relu5_1'], W=self.get_weights(vgg_layers, 30))
    net['relu5_2'] = self.relu_layer('relu5_2', net['conv5_2'], b=self.get_bias(vgg_layers, 30))

    net['conv5_3'] = self.conv_layer('conv5_3', net['relu5_2'], W=self.get_weights(vgg_layers, 32))
    net['relu5_3'] = self.relu_layer('relu5_3', net['conv5_3'], b=self.get_bias(vgg_layers, 32))

    net['conv5_4'] = self.conv_layer('conv5_4', net['relu5_3'], W=self.get_weights(vgg_layers, 34))
    net['relu5_4'] = self.relu_layer('relu5_4', net['conv5_4'], b=self.get_bias(vgg_layers, 34))

    net['pool5']   = self.pool_layer('pool5', net['relu5_4'])

    return net

  def conv_layer(self,layer_name, layer_input, W):
    conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='SAME')
    if self.verbose: print('--{} | shape={} | weights_shape={}'.format(layer_name, 
      conv.get_shape(), W.get_shape()))
    return conv

  def relu_layer(self,layer_name, layer_input, b):
    relu = tf.nn.relu(layer_input + b)
    if self.verbose: 
      print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.get_shape(), 
        b.get_shape()))
    return relu

  def pool_layer(self,layer_name, layer_input):
    if self.pooling_type == 'avg':
      pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], 
        strides=[1, 2, 2, 1], padding='SAME')
    elif self.pooling_type == 'max':
      pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1], 
        strides=[1, 2, 2, 1], padding='SAME')
    if self.verbose: 
      print('--{}   | shape={}'.format(layer_name, pool.get_shape()))
    return pool

  def get_weights(self,vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    W = tf.constant(weights)
    return W

  def get_bias(self,vgg_layers, i):
    bias = vgg_layers[i][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    return b

  def content_layer_loss(self,p, x):
    _, h, w, d = p.get_shape()
    M = h.value * w.value
    N = d.value
    if self.content_loss_function   == 1:
      K = 1. / (2. * N**0.5 * M**0.5)
    elif self.content_loss_function == 2:
      K = 1. / (N * M)
    elif self.content_loss_function == 3:  
      K = 1. / 2.
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss

  def style_layer_loss(self,a, x):
    _, h, w, d = a.get_shape()
    M = h.value * w.value
    N = d.value
    A = self.gram_matrix(a, M, N)
    G = self.gram_matrix(x, M, N)
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

  def gram_matrix(self,x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G

  def mask_style_layer(self,a, x, mask_img):
    _, h, w, d = a.get_shape()
    mask = get_mask_image(mask_img, w.value, h.value)
    mask = tf.convert_to_tensor(mask)
    tensors = []
    for _ in range(d.value): 
      tensors.append(mask)
    mask = tf.stack(tensors, axis=2)
    mask = tf.stack(mask, axis=0)
    mask = tf.expand_dims(mask, 0)
    a = tf.multiply(a, mask)
    x = tf.multiply(x, mask)
    return a, x

  def sum_masked_style_losses(self,sess, net, style_imgs):
    total_style_loss = 0.
    weights = style_imgs_weights
    # masks = args.style_mask_imgs
    for img, img_weight, img_mask in zip(style_imgs, weights, masks):
      sess.run(net['input'].assign(img))
      style_loss = 0.
      for layer, weight in zip(style_layers, style_layer_weights):
        a = sess.run(net[layer])
        x = net[layer]
        a = tf.convert_to_tensor(a)
        a, x = mask_style_layer(a, x, img_mask)
        style_loss += style_layer_loss(a, x) * weight
      style_loss /= float(len(style_layers))
      total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss

  def sum_style_losses(self,sess, net, style_imgs):
    total_style_loss = 0.
    weights = self.style_imgs_weights
    for img, img_weight in zip(style_imgs, weights):
      sess.run(net['input'].assign(img))
      style_loss = 0.
      for layer, weight in zip(self.style_layers, self.style_layer_weights):
        a = sess.run(net[layer])
        x = net[layer]
        a = tf.convert_to_tensor(a)
        style_loss += self.style_layer_loss(a, x) * weight
      style_loss /= float(len(self.style_layers))
      total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss

  def sum_content_losses(self,sess, net, content_img):
    sess.run(net['input'].assign(content_img))
    content_loss = 0.
    for layer, weight in zip(self.content_layers, self.content_layer_weights):
      p = sess.run(net[layer])
      x = net[layer]
      p = tf.convert_to_tensor(p)
      content_loss += self.content_layer_loss(p, x) * weight
    content_loss /= float(len(self.content_layers))
    return content_loss

  def temporal_loss(self,x, w, c):
    c = c[np.newaxis,:,:,:]
    D = float(x.size)
    loss = (1. / D) * tf.reduce_sum(c * tf.nn.l2_loss(x - w))
    loss = tf.cast(loss, tf.float32)
    return loss

  def get_longterm_weights(self,i, j):
    c_sum = 0.
    for k in range(prev_frame_indices):
      if i - k > i - j:
        c_sum += get_content_weights(i, i - k)
    c = get_content_weights(i, i - j)
    c_max = tf.maximum(c - c_sum, 0.)
    return c_max

  def sum_longterm_temporal_losses(self,sess, net, frame, input_img):
    x = sess.run(net['input'].assign(input_img))
    loss = 0.
    for j in range(prev_frame_indices):
      prev_frame = frame - j
      w = get_prev_warped_frame(frame)
      c = get_longterm_weights(frame, prev_frame)
      loss += temporal_loss(x, w, c)
    return loss

  def sum_shortterm_temporal_losses(self,sess, net, frame, input_img):
    x = sess.run(net['input'].assign(input_img))
    prev_frame = frame - 1
    w = get_prev_warped_frame(frame)
    c = get_content_weights(frame, prev_frame)
    loss = temporal_loss(x, w, c)
    return loss

  def preprocess(self,img):
    imgpre = np.copy(img)
    # bgr to rgb
    imgpre = imgpre[...,::-1]
    # shape (h, w, d) to (1, h, w, d)
    imgpre = imgpre[np.newaxis,:,:,:]
    imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    return imgpre

  def postprocess(self,img):
    imgpost = np.copy(img)
    imgpost += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    # shape (1, h, w, d) to (h, w, d)
    imgpost = imgpost[0]
    imgpost = np.clip(imgpost, 0, 255).astype('uint8')
    # rgb to bgr
    # imgpost = imgpost[...,::-1]
    return imgpost

  def read_weights_file(self,path):
    lines = open(path).readlines()
    header = list(map(int, lines[0].split(' ')))
    w = header[0]
    h = header[1]
    vals = np.zeros((h, w), dtype=np.float32)
    for i in range(1, len(lines)):
      line = lines[i].rstrip().split(' ')
      vals[i-1] = np.array(list(map(np.float32, line)))
      vals[i-1] = list(map(lambda x: 0. if x < 255. else 1., vals[i-1]))
    # expand to 3 channels
    weights = np.dstack([vals.astype(np.float32)] * 3)
    return weights

  def normalize(self, weights):
    denom = sum(weights)
    if denom > 0.:
      return [float(i) / denom for i in weights]
    else: return [0.] * len(weights)

  def maybe_make_directory(self,dir_path):
    if not os.path.exists(dir_path):  
      os.makedirs(dir_path)

  def check_image(self,img, path):
    if img is None:
      raise OSError(errno.ENOENT, "No such file", path)

  '''
    rendering -- where the magic happens
  '''
  def stylize(self,content_img, style_imgs, init_img, frame=None):
    with tf.device(self.device_opts), tf.Session() as sess:
      # setup network
      net = self.build_model(content_img)
      
      # style loss
      L_style = self.sum_style_losses(sess, net, style_imgs)
      
      # content loss
      L_content = self.sum_content_losses(sess, net, content_img)
      
      # denoising loss
      L_tv = tf.image.total_variation(net['input'])
      
      # loss weights
      alpha = self.content_weight
      beta  = self.style_weight
      theta = self.tv_weight
      
      # total loss
      L_total  = alpha * L_content
      L_total += beta  * L_style
      L_total += theta * L_tv
      
      # video temporal loss
      #if video and frame > 1:
      #  gamma      = arguments.temporal_weight
      #  L_temporal = sum_shortterm_temporal_losses(sess, net, frame, init_img)
      #  L_total   += gamma * L_temporal

      # optimization algorithm
      optimizer = self.get_optimizer(L_total)

      if self.optimizer_type == 'adam':
        self.minimize_with_adam(sess, net, optimizer, init_img, L_total)
      elif self.optimizer_type == 'lbfgs':
        self.minimize_with_lbfgs(sess, net, optimizer, init_img)
      
      output_img = sess.run(net['input'])
      return output_img
      
      # if args.original_colors:
      #   output_img = convert_to_original_colors(np.copy(content_img), output_img)

      # if args.video:
      #   write_video_output(frame, output_img)
      # else:
      #   write_image_output(output_img, content_img, style_imgs, init_img)

  def minimize_with_lbfgs(self,sess, net, optimizer, init_img):
    if self.verbose: print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)

  def minimize_with_adam(self,sess, net, optimizer, init_img, loss):
    if self.verbose: print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')
    train_op = optimizer.minimize(loss)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    iterations = 0
    while (iterations < self.max_iterations):
      sess.run(train_op)
      if iterations % self.print_iterations == 0 and self.verbose:
        curr_loss = loss.eval()
        print("At iterate {}\tf=  {}".format(iterations, curr_loss))
      iterations += 1

  def get_optimizer(self,loss):
    self.print_iterations = 50 if self.verbose else 0
    if self.optimizer_type == 'lbfgs':
      optimizer = tf.contrib.opt.ScipyOptimizerInterface(
        loss, method='L-BFGS-B',
        options={'maxiter': self.max_iterations,
                    'disp': self.print_iterations})
    elif self.optimizer_type == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer

  def get_init_image(self,init_type, content_img, style_imgs, frame=None):
    if init_type == 'content':
      return content_img
    elif init_type == 'style':
      return style_imgs[0]
    elif init_type == 'random':
      init_img = get_noise_image(noise_ratio, content_img)
      return init_img
    # only for video frames
    elif init_type == 'prev':
      init_img = get_prev_frame(frame)
      return init_img
    elif init_type == 'prev_warped':
      init_img = get_prev_warped_frame(frame)
      return init_img

  def get_content_weights(self,frame, prev_frame):
    forward_fn = content_weights_frmt.format(str(prev_frame), str(frame))
    backward_fn = content_weights_frmt.format(str(frame), str(prev_frame))
    forward_path = os.path.join(video_input_dir, forward_fn)
    backward_path = os.path.join(video_input_dir, backward_fn)
    forward_weights = read_weights_file(forward_path)
    backward_weights = read_weights_file(backward_path)
    return forward_weights #, backward_weights

  def convert_to_original_colors(self,content_img, stylized_img):
    content_img  = postprocess(content_img)
    stylized_img = postprocess(stylized_img)
    if color_convert_type == 'yuv':
      cvt_type = cv2.COLOR_BGR2YUV
      inv_cvt_type = cv2.COLOR_YUV2BGR
    elif color_convert_type == 'ycrcb':
      cvt_type = cv2.COLOR_BGR2YCR_CB
      inv_cvt_type = cv2.COLOR_YCR_CB2BGR
    elif color_convert_type == 'luv':
      cvt_type = cv2.COLOR_BGR2LUV
      inv_cvt_type = cv2.COLOR_LUV2BGR
    elif color_convert_type == 'lab':
      cvt_type = cv2.COLOR_BGR2LAB
      inv_cvt_type = cv2.COLOR_LAB2BGR
    content_cvt = cv2.cvtColor(content_img, cvt_type)
    stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)
    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
    dst = preprocess(dst)
    return dst

  def render_single_image(self,content_img,style_imgs):
    with tf.Graph().as_default():
      print('\n---- RENDERING SINGLE IMAGE ----\n')
      init_img = self.get_init_image(self.init_img_type, content_img, style_imgs)
      tick = time.time()
      output_img = self.stylize(content_img, style_imgs, init_img)
      tock = time.time()
      print('Single image elapsed time: {}'.format(tock - tick))
      return output_img

  # def main():
  #   global args, content_img
  #   # args = parse_args()
  #   # if args.video: render_video()
  #   render_single_image()

  # if __name__ == '__main__':
  #   main()

  def set_variables():
    verbose = True
    
    device_opts = '/gpu:0' #'/cpu:0'
