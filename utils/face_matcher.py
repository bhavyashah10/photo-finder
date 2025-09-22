import cv2
import face_recognition
import numpy as np
import os
from PIL import Image, ExifTags
import pickle
import json
from datetime import datetime

class FaceMatcher:
    """
    A utility class for face detection, encoding, and matching in photos.
    """
    
    def __init__(self, tolerance=0.45, model='cnn', min_confidence=0.55):
        """
        Initialize the FaceMatcher with more strict settings.
        
        Args:
            tolerance (float): How strict the face matching should be. Lower is more strict.
                              Recommended: 0.4-0.5 for strict matching, 0.6 for loose matching
            model (str): Face detection model to use ('hog' for CPU, 'cnn' for GPU)
            min_confidence (float): Minimum confidence score to accept a match (0.0-1.0)
        """
        self.tolerance = tolerance
        self.model = model
        self.min_confidence = min_confidence  # Add minimum confidence threshold
        self.encodings_cache = {}
        self.cache_file = 'face_encodings_cache.pkl'
        self.load_cache()
    
    def load_cache(self):
        """Load previously computed face encodings from cache file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.encodings_cache = pickle.load(f)
                print(f"Loaded {len(self.encodings_cache)} cached encodings")
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.encodings_cache = {}
    
    def save_cache(self):
        """Save computed face encodings to cache file for faster future processing."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.encodings_cache, f)
            print(f"Saved {len(self.encodings_cache)} encodings to cache")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def fix_image_orientation(self, image_path):
        """
        Fix image orientation based on EXIF data.
        Many phone photos have rotation info in EXIF that needs to be applied.
        """
        try:
            image = Image.open(image_path)
            
            # Check if image has EXIF data
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif is not None:
                    # Look for orientation tag
                    for tag, value in exif.items():
                        decoded = ExifTags.TAGS.get(tag, tag)
                        if decoded == 'Orientation':
                            if value == 3:
                                image = image.rotate(180, expand=True)
                            elif value == 6:
                                image = image.rotate(270, expand=True)
                            elif value == 8:
                                image = image.rotate(90, expand=True)
                            break
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return np.array(image)
        except Exception as e:
            print(f"Error fixing orientation for {image_path}: {e}")
            # Fallback to basic loading
            return face_recognition.load_image_file(image_path)
    
    def preprocess_image(self, image_path, max_size=1024):
        """
        Preprocess image for better face recognition:
        - Fix orientation
        - Resize if too large
        - Enhance contrast if needed
        """
        try:
            # Load and fix orientation
            image = self.fix_image_orientation(image_path)
            
            # Resize if image is too large (for faster processing)
            height, width = image.shape[:2]
            if max(height, width) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int((height * max_size) / width)
                else:
                    new_height = max_size
                    new_width = int((width * max_size) / height)
                
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return image
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return face_recognition.load_image_file(image_path)
    
    def get_face_encodings(self, image_path, use_cache=True):
        """
        Extract face encodings from an image with caching and preprocessing.
        
        Args:
            image_path (str): Path to the image file
            use_cache (bool): Whether to use/update cache
            
        Returns:
            list: List of face encodings found in the image
        """
        # Check cache first
        if use_cache and image_path in self.encodings_cache:
            cache_entry = self.encodings_cache[image_path]
            # Check if file has been modified since caching
            file_mod_time = os.path.getmtime(image_path)
            if cache_entry['timestamp'] >= file_mod_time:
                return cache_entry['encodings']
        
        try:
            # Preprocess the image
            image = self.preprocess_image(image_path)
            
            # Find face locations first
            face_locations = face_recognition.face_locations(image, model=self.model)
            
            if not face_locations:
                print(f"No faces found in {image_path}")
                return []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # Cache the results
            if use_cache:
                self.encodings_cache[image_path] = {
                    'encodings': face_encodings,
                    'timestamp': os.path.getmtime(image_path),
                    'face_count': len(face_encodings)
                }
            
            print(f"Found {len(face_encodings)} face(s) in {os.path.basename(image_path)}")
            return face_encodings
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []
    
    def find_matching_photos(self, guest_photo_path, wedding_photos_folder, allowed_extensions=None):
        """
        Find all wedding photos that contain the guest's face with improved filtering.
        """
        if allowed_extensions is None:
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        
        # Get guest face encodings
        guest_encodings = self.get_face_encodings(guest_photo_path)
        
        if not guest_encodings:
            return {
                'success': False,
                'error': 'No face detected in the guest photo. Please upload a clear photo with a visible face.',
                'matches': [],
                'stats': {}
            }
        
        # Use the first (and presumably primary) face encoding
        guest_encoding = guest_encodings[0]
        
        matches = []
        processed_count = 0
        error_count = 0
        total_faces_found = 0
        rejected_low_confidence = 0
        
        # Get all wedding photo files
        wedding_files = [f for f in os.listdir(wedding_photos_folder) 
                        if f.lower().split('.')[-1] in allowed_extensions]
        
        print(f"Processing {len(wedding_files)} wedding photos...")
        print(f"Using tolerance: {self.tolerance}, min_confidence: {self.min_confidence}")
        
        for filename in wedding_files:
            try:
                photo_path = os.path.join(wedding_photos_folder, filename)
                wedding_encodings = self.get_face_encodings(photo_path)
                
                processed_count += 1
                total_faces_found += len(wedding_encodings)
                
                # Check each face in the wedding photo
                best_match_confidence = 0
                best_match_distance = float('inf')
                
                for i, wedding_encoding in enumerate(wedding_encodings):
                    # Compare faces
                    face_distances = face_recognition.face_distance([wedding_encoding], guest_encoding)
                    face_distance = face_distances[0]
                    confidence = 1 - face_distance
                    
                    # Check both tolerance and minimum confidence
                    if face_distance <= self.tolerance and confidence >= self.min_confidence:
                        # Keep track of the best match in this photo
                        if confidence > best_match_confidence:
                            best_match_confidence = confidence
                            best_match_distance = face_distance
                
                # Only add the photo if we found a good match
                if best_match_confidence > 0:
                    matches.append({
                        'filename': filename,
                        'path': f'/static/uploads/wedding_photos/{filename}',
                        'confidence': float(best_match_confidence),
                        'face_distance': float(best_match_distance)
                    })
                    print(f"✓ Strong match in {filename} (confidence: {best_match_confidence:.3f}, distance: {best_match_distance:.3f})")
                else:
                    # Check if there were faces but they didn't meet our criteria
                    for wedding_encoding in wedding_encodings:
                        face_distance = face_recognition.face_distance([wedding_encoding], guest_encoding)[0]
                        if face_distance <= self.tolerance:  # Met tolerance but not confidence
                            rejected_low_confidence += 1
                            print(f"✗ Rejected {filename} - low confidence: {1-face_distance:.3f}")
                            break
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                error_count += 1
                continue
        
        # Save updated cache
        self.save_cache()
        
        # Sort matches by confidence (highest first)
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        stats = {
            'total_photos_processed': processed_count,
            'total_faces_found': total_faces_found,
            'rejected_low_confidence': rejected_low_confidence,
            'errors': error_count,
            'average_confidence': sum(m['confidence'] for m in matches) / len(matches) if matches else 0,
            'tolerance_used': self.tolerance,
            'min_confidence_used': self.min_confidence
        }
        
        print(f"\nResults: {len(matches)} matches found, {rejected_low_confidence} rejected for low confidence")
        if matches:
            print(f"Confidence range: {min(m['confidence'] for m in matches):.3f} - {max(m['confidence'] for m in matches):.3f}")
        
        return {
            'success': True,
            'matches': matches,
            'total_matches': len(matches),
            'stats': stats
        }
    
    def batch_process_wedding_photos(self, wedding_photos_folder, allowed_extensions=None):
        """
        Pre-process all wedding photos to build face encoding cache.
        This can be run after photographer uploads to speed up guest searches.
        
        Args:
            wedding_photos_folder (str): Folder containing wedding photos
            allowed_extensions (set): Set of allowed file extensions
            
        Returns:
            dict: Processing statistics
        """
        if allowed_extensions is None:
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        
        wedding_files = [f for f in os.listdir(wedding_photos_folder) 
                        if f.lower().split('.')[-1] in allowed_extensions]
        
        processed = 0
        errors = 0
        total_faces = 0
        
        print(f"Batch processing {len(wedding_files)} wedding photos...")
        
        for filename in wedding_files:
            try:
                photo_path = os.path.join(wedding_photos_folder, filename)
                encodings = self.get_face_encodings(photo_path, use_cache=True)
                total_faces += len(encodings)
                processed += 1
                
                if processed % 10 == 0:
                    print(f"Processed {processed}/{len(wedding_files)} photos...")
                    
            except Exception as e:
                print(f"Error batch processing {filename}: {e}")
                errors += 1
        
        # Save cache after batch processing
        self.save_cache()
        
        return {
            'total_files': len(wedding_files),
            'processed': processed,
            'errors': errors,
            'total_faces_found': total_faces,
            'cache_size': len(self.encodings_cache)
        }
    
    def get_face_locations_with_confidence(self, image_path):
        """
        Get face locations along with confidence scores.
        Useful for debugging and showing face detection results.
        
        Returns:
            list: List of dictionaries with face location and confidence info
        """
        try:
            image = self.preprocess_image(image_path)
            face_locations = face_recognition.face_locations(image, model=self.model)
            
            faces_info = []
            for i, (top, right, bottom, left) in enumerate(face_locations):
                faces_info.append({
                    'face_id': i,
                    'location': {
                        'top': top,
                        'right': right,
                        'bottom': bottom,
                        'left': left
                    },
                    'width': right - left,
                    'height': bottom - top
                })
            
            return faces_info
            
        except Exception as e:
            print(f"Error getting face locations for {image_path}: {e}")
            return []
    
    def clear_cache(self):
        """Clear the face encodings cache."""
        self.encodings_cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("Face encodings cache cleared")
    
    def get_cache_stats(self):
        """Get statistics about the current cache."""
        total_encodings = sum(entry.get('face_count', 0) for entry in self.encodings_cache.values())
        return {
            'cached_images': len(self.encodings_cache),
            'total_encodings': total_encodings,
            'cache_file_exists': os.path.exists(self.cache_file)
        }