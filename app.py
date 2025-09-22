from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import sys

# Add utils folder to path so we can import our face_matcher
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from face_matcher import FaceMatcher

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize face matcher
face_matcher = FaceMatcher(tolerance=0.45, model='cnn', min_confidence=0.55) # Use 'cnn' if you have GPU

# Create upload directories
UPLOAD_FOLDER = 'static/uploads'
WEDDING_PHOTOS_FOLDER = os.path.join(UPLOAD_FOLDER, 'wedding_photos')
GUEST_PHOTOS_FOLDER = os.path.join(UPLOAD_FOLDER, 'guest_photos')

for folder in [WEDDING_PHOTOS_FOLDER, GUEST_PHOTOS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/photographer')
def photographer():
    # Get count of uploaded wedding photos
    photo_count = len([f for f in os.listdir(WEDDING_PHOTOS_FOLDER) if allowed_file(f)])
    return render_template('photographer.html', photo_count=photo_count)

@app.route('/guest')
def guest():
    return render_template('guest.html')

@app.route('/upload_wedding_photos', methods=['POST'])
def upload_wedding_photos():
    if 'photos' not in request.files:
        return jsonify({'error': 'No photos provided'}), 400
    
    files = request.files.getlist('photos')
    uploaded_count = 0
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid duplicates
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            
            file_path = os.path.join(WEDDING_PHOTOS_FOLDER, filename)
            file.save(file_path)
            uploaded_count += 1
    
    return jsonify({
        'success': True, 
        'message': f'Successfully uploaded {uploaded_count} photos'
    })

@app.route('/preprocess_photos', methods=['POST'])
def preprocess_photos():
    """
    Endpoint to pre-process all wedding photos and build face encoding cache.
    This can be called after photographer uploads to speed up guest searches.
    """
    try:
        stats = face_matcher.batch_process_wedding_photos(WEDDING_PHOTOS_FOLDER, ALLOWED_EXTENSIONS)
        return jsonify({
            'success': True,
            'message': f'Pre-processed {stats["processed"]} photos, found {stats["total_faces_found"]} faces',
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error during pre-processing: {str(e)}'
        }), 500

@app.route('/find_matches', methods=['POST'])
def find_matches():
    if 'guest_photo' not in request.files:
        return jsonify({'error': 'No guest photo provided'}), 400
    
    file = request.files['guest_photo']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400
    
    # Save guest photo temporarily
    filename = secure_filename(file.filename)
    temp_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    guest_photo_path = os.path.join(GUEST_PHOTOS_FOLDER, temp_filename)
    file.save(guest_photo_path)
    
    try:
        # Use the FaceMatcher to find matches
        result = face_matcher.find_matching_photos(
            guest_photo_path, 
            WEDDING_PHOTOS_FOLDER, 
            ALLOWED_EXTENSIONS
        )
        
        # Clean up temporary guest photo
        if os.path.exists(guest_photo_path):
            os.remove(guest_photo_path)
        
        if result['success']:
            return jsonify({
                'success': True,
                'matches': result['matches'],
                'total_matches': result['total_matches'],
                'stats': result['stats']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
    except Exception as e:
        # Clean up temporary guest photo
        if os.path.exists(guest_photo_path):
            os.remove(guest_photo_path)
        return jsonify({'error': f'Error processing photo: {str(e)}'}), 500

@app.route('/get_wedding_photos')
def get_wedding_photos():
    """Get list of all wedding photos for photographer view"""
    photos = []
    for filename in os.listdir(WEDDING_PHOTOS_FOLDER):
        if allowed_file(filename):
            photos.append({
                'filename': filename,
                'path': f'/static/uploads/wedding_photos/{filename}'
            })
    return jsonify(photos)

@app.route('/get_cache_stats')
def get_cache_stats():
    """Get face recognition cache statistics"""
    try:
        stats = face_matcher.get_cache_stats()
        return jsonify({
            'success': True,
            'cache_stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the face recognition cache"""
    try:
        face_matcher.clear_cache()
        return jsonify({
            'success': True,
            'message': 'Face recognition cache cleared successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/face_debug/<path:filename>')
def face_debug(filename):
    """Debug endpoint to see face detection results for a specific image"""
    try:
        image_path = os.path.join(WEDDING_PHOTOS_FOLDER, filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404
        
        faces_info = face_matcher.get_face_locations_with_confidence(image_path)
        encodings = face_matcher.get_face_encodings(image_path)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'faces_detected': len(faces_info),
            'faces_info': faces_info,
            'encodings_count': len(encodings)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Wedding Photo Finder...")
    print(f"Face matcher initialized with tolerance: {face_matcher.tolerance}")
    print(f"Upload folders: {WEDDING_PHOTOS_FOLDER}, {GUEST_PHOTOS_FOLDER}")
    
    app.run(debug=True, host='0.0.0.0', port=5001)