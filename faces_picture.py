from PIL import Image, ImageDraw
import face_recognition
import os

# Const directory for training
train_dir = "dataset/train"

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

for file in os.listdir(train_dir):
    # Load and encode each images of the train directory
    trained_image = face_recognition.load_image_file(train_dir + '/' + file)
    try:
        trained_image_encoding = face_recognition.face_encodings(trained_image)[0]
    except IndexError:
        print(file + " rejected")
        continue

    # Add them to array for compare later
    known_face_encodings.append(trained_image_encoding)
    known_face_names.append(file)

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("dataset/test/mbds.jpg")

# Find all the faces in the image using the default HOG-based model.
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

print("I found {} face(s) in this photograph.".format(len(face_locations)))

# Create a Pillow ImageDraw Draw instance to draw with
pil_image = Image.fromarray(image)
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 0, 0), outline=(255, 0, 0))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()