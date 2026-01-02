import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import os
import matplotlib.pyplot as plt
import seaborn as sns # We need this for the plot

# --- 0. CONFIGURATION ---
# We will resize all frames to 128x128. This is faster.
IMG_SIZE = 128
# We'll train for 10 "epochs" (passes over the data).
EPOCHS = 10
BATCH_SIZE = 32

# --- 1. DEFINE VIDEO PATHS ---
# !!! IMPORTANT !!!
# Change these two lines to match your exact filenames in the /data/ folder
NORMAL_VIDEO_NAME = "test_video.mp4"
TEST_VIDEO_NAME = "suspicious.mp4"
# !!! END IMPORTANT !!!

# --- These paths are automatic ---
NORMAL_VIDEO_PATH = os.path.join('../data', NORMAL_VIDEO_NAME)
TEST_VIDEO_PATH = os.path.join('../data', TEST_VIDEO_NAME)
MODEL_SAVE_PATH = '../models/autoencoder_model.h5'
PLOT_SAVE_PATH = '../outputs/anomaly_threshold_plot.png'
VIDEO_OUT_PATH = os.path.join('../outputs', f'anomaly_output_{TEST_VIDEO_NAME}')

# --- 2. HELPER FUNCTION: VIDEO FRAME LOADER ---
# This function will read a video and extract its frames
def load_video_frames(video_path, img_size):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    print(f"Loading frames from {video_path}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale (simpler for the model)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to our standard 128x128 size
        resized_frame = cv2.resize(gray_frame, (img_size, img_size))
        # Normalize the pixel values to be between 0 and 1
        normalized_frame = resized_frame.astype('float32') / 255.0
        frames.append(normalized_frame)
    
    cap.release()
    print(f"Loaded {len(frames)} frames.")
    # Convert list to a 4D NumPy array: (num_frames, height, width, 1 channel)
    return np.array(frames).reshape(-1, img_size, img_size, 1)

# --- 3. DEFINE THE AUTOENCODER MODEL ---
def build_autoencoder(img_size):
    input_img = Input(shape=(img_size, img_size, 1))

    # Encoder (compresses the image)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder (reconstructs the image from the compressed form)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse') # Mean Squared Error
    return autoencoder

# --- 4. PART 1: TRAINING ---
print("--- PART 1: TRAINING AUTOENCODER ---")
# Load the "normal" frames for training
X_train = load_video_frames(NORMAL_VIDEO_PATH, IMG_SIZE)

if X_train is not None and len(X_train) > 0:
    # Build the model
    autoencoder = build_autoencoder(IMG_SIZE)
    autoencoder.summary()
    
    # Train the model. Notice X and y are the same (X_train)!
    # The model learns to reconstruct its own input.
    print("Training autoencoder...")
    autoencoder.fit(X_train, X_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_split=0.1)
    
    # Save the trained model
    autoencoder.save(MODEL_SAVE_PATH)
    print(f"Autoencoder model saved to {MODEL_SAVE_PATH}")

    # --- 5. PART 2: ANOMALY THRESHOLDING ---
    print("\n--- PART 2: FINDING ANOMALY THRESHOLD ---")
    # We need to find out what a "normal" error looks like.
    reconstructed_imgs = autoencoder.predict(X_train)
    # Calculate the error (loss) for each training frame
    train_loss = tf.keras.losses.mse(reconstructed_imgs, X_train)
    frame_losses = np.mean(train_loss, axis=(1, 2))
    
    # Plot a graph of these normal errors
    plt.figure()
    sns.histplot(frame_losses, bins=50, kde=True)
    plt.title('Reconstruction Error for Normal Frames')
    plt.xlabel('Mean Squared Error (Loss)')
    plt.ylabel('Frequency')
    
    # We set our "anomaly threshold" high.
    # We'll say any error in the top 2% (0.98) must be an anomaly.
    anomaly_threshold = np.quantile(frame_losses, 0.95) 
    plt.axvline(anomaly_threshold, color='r', linestyle='--', label=f'Threshold = {anomaly_threshold:.4f}')
    plt.legend()
    plt.savefig(PLOT_SAVE_PATH)
    print(f"Anomaly threshold set to: {anomaly_threshold:.4f}")
    print(f"Threshold plot saved to {PLOT_SAVE_PATH}")

    # --- 6. PART 3: INFERENCE (TESTING ON THE SUSPICIOUS VIDEO) ---
    print("\n--- PART 3: DETECTING ANOMALIES IN TEST VIDEO ---")
    
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open test video {TEST_VIDEO_PATH}")
        exit()

    # Get video properties for the new output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (frame_width, frame_height))

    print("Processing test video for anomalies...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Pre-process the frame just like we did for training
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (IMG_SIZE, IMG_SIZE))
        normalized_frame = resized_frame.astype('float32') / 255.0
        frame_input = normalized_frame.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # Get the reconstruction from our trained model
        reconstructed_img = autoencoder.predict(frame_input)
        
        # Calculate this frame's reconstruction error
        frame_loss = np.mean(tf.keras.losses.mse(reconstructed_img, frame_input))
        
        # Check if the error is HIGHER than our threshold
        if frame_loss > anomaly_threshold:
            # It's an anomaly! Draw red text on the original frame
            cv2.putText(frame, "ANOMALY DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Write the frame to the new video
        out.write(frame)

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nAnomaly detection complete! Output video saved to: {VIDEO_OUT_PATH}")

else:
    print("Training failed. Please check the path to your 'normal' video.")