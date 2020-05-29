# Import the Runway SDK. Please install it first with
# `pip install runway-python`.
import runway
from runway.data_types import number, text, image, boolean, category
from neural_style import NeuralStyle

#setup_options = {
#    'seed': number(min=0, max=10000, step=1, default=101, description='Seed for the random number generator.'),
#}
#@runway.setup(options=setup_options)
def setup():
    model = NeuralStyle()
    return model

input_list = {
    'content_image': image,
    'style_image_1': image,
    'original_colors': boolean(default=False),
    'style_only': boolean(default=False),
    'max_iterations': number(min=50, max=1500, step=50, default=500, description='Iterations'),
    'content_layer': category(choices=['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1', 'conv3_2', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4','conv5_1','conv5_2'], default='conv4_2', description='what VGG19 layer to use for content'),
    'style_scale': number(min=0.1, max=2.0, step=.05, default=1.0, description='Scale of style images.'),
}

@runway.command(name='generate',
                inputs=input_list,
                outputs={ 'image': image() },
                description='Neural Style Transfer: Transfer the style from one image to the content of another.')
def generate(model, args):
    print(args)

    model_new = NeuralStyle()
    model_new.content_layers = [args['content_layer']]

    # print('[GENERATE] Ran with content image: "{}"'.format(args['content_image']))
    # print('[GENERATE] Ran with first style image: "{}"'.format(args['style_image_1']))
    # # print('[GENERATE] Ran with second style image: "{}"'.format(args['style_image_2']))
    # Generate a PIL or Numpy image based on the input caption, and return it
    output_image = model_new.run(
        args['content_image'],
        args['style_image_1'],
        args['original_colors'],
        args['max_iterations'],
        args['style_scale'],
        args['style_only']
        )
    return {
        'image':output_image
    }

if __name__ == '__main__':
    runway.run(port=8000, debug=True)
