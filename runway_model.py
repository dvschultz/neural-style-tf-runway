# Import the Runway SDK. Please install it first with
# `pip install runway-python`.
import runway
from runway.data_types import number, text, image,boolean
from neural_style import NeuralStyle

setup_options = {
    'seed': number(min=0, max=10000, step=1, default=101, description='Seed for the random number generator.'),
}
@runway.setup(options=setup_options)
def setup(opts):
    model = NeuralStyle(opts)
    return model

input_list = {
    'content_image': image,
    'style_image_1': image,
    'original_colors': boolean(default=False),
    'max_iterations': number(min=50, max=1500, step=50, default=500, description='Iterations'),
    'max_size': number(min=256, max=1600, step=16, default=1200, description='Maximum output size.'),
    'style_scale': number(min=0.1, max=2.0, step=.05, default=1.0, description='Scale of style images.'),
}

@runway.command(name='generate',
                inputs=input_list,
                outputs={ 'image': image() },
                description='Neural Style Transfer: Transfer the style from one image to the content of another.')
def generate(model, args):
    print(args)
    # print('[GENERATE] Ran with content image: "{}"'.format(args['content_image']))
    # print('[GENERATE] Ran with first style image: "{}"'.format(args['style_image_1']))
    # # print('[GENERATE] Ran with second style image: "{}"'.format(args['style_image_2']))
    # Generate a PIL or Numpy image based on the input caption, and return it
    output_image = model.run(
        args['content_image'],
        args['style_image_1'],
        args['original_colors'],
        args['max_iterations'],
        args['max_size'],
        args['style_scale'],
        )
    return {
        'image':output_image
    }

if __name__ == '__main__':
    runway.run(port=8000, debug=True)
