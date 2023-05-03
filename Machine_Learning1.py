import io

from google.cloud import vision
from google.cloud.vision_v1 import types

# Instantiates a client
client = vision.ImageAnnotatorClient()

# Loads the image file into memory
with io.open('IELTS-template.jpg', 'rb') as image_file:
    content = image_file.read()
image = types.Image(content=content)

# Performs label detection on the image file
response = client.annotate_image({
    'image': image,
    'features': [
        {'type': vision.enums.Feature.Type.FACE_DETECTION},
        {'type': vision.enums.Feature.Type.TEXT_DETECTION},
        {'type': vision.enums.Feature.Type.OBJECT_LOCALIZATION},
    ],
})

# Prints the results
print('Faces:')
for face in response.face_annotations:
    print(' - Joy likelihood: {}'.format(face.joy_likelihood.name))

print('Text:')
for text in response.text_annotations:
    print('\n"{}"'.format(text.description))

print('Objects:')
for obj in response.localized_object_annotations:
    print('{} (confidence: {})'.format(obj.name, obj.score))
