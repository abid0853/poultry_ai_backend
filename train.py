import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# --- 1. SETUP PATHS ---
# Update these to point to your actual dataset folders
TRAIN_DIR = './dataset/train' 
VALID_DIR = './dataset/valid'

# --- 2. LOAD & PREPARE DATA ---
def load_data(directory):
    # Read the CSV file inside the folder
    csv_path = os.path.join(directory, 'annotations.csv')
    df = pd.read_csv(csv_path)
    
    # ASSUMPTION: Your CSV has columns like 'filename' and 'class'
    # If your columns are named differently (e.g., 'image_id', 'label'), change them here!
    # We only keep the filename and the label (Healthy/Sick)
    return df

train_df = load_data(TRAIN_DIR)
valid_df = load_data(VALID_DIR)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(valid_df)}")

# --- 3. CREATE IMAGE GENERATORS ---
# This scales pixel values from 0-255 to 0-1 (helps AI learn faster)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Connect the CSV data to the actual Images
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col='filename',      # The column name in your CSV with image names
    y_col='class',         # The column name in your CSV with labels (e.g. 'sick', 'healthy')
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical' # Detects multiple classes (Healthy vs Sick)
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory=VALID_DIR,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# --- 4. BUILD THE MODEL (MobileNetV2) ---
# We use a pre-trained model (Transfer Learning) to get high accuracy with less data
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model so we don't ruin its pre-learned features
base_model.trainable = False

# Add our custom layers for "Chicken Health"
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x) # 2 outputs: Healthy or Sick

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 5. TRAIN ---
print("Starting training...")
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10  # Increase this if accuracy is low
)

# --- 6. SAVE THE MODEL ---
model.save('poultry_disease_model.h5')
print("Model saved as poultry_disease_model.h5")

# Save the class mapping so we know 0=Healthy, 1=Sick
print("Class Indices:", train_generator.class_indices)