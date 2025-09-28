# Photo Finder

A web application that uses AI face recognition to help guests find all photos of themselves from the collection. Upload a selfie, and the system will automatically find and return all photos containing your face!

## ✨ Features

- **Photographer Portal**: Upload and manage all photos
- **Guest Portal**: Upload a reference photo to find all matching photos
- **AI Face Recognition**: Advanced face detection and matching using deep learning
- **Smart Caching**: Optimized performance with face encoding caching
- **Batch Processing**: Pre-process photos for faster guest searches
- **Confidence Scoring**: Adjustable matching strictness and confidence thresholds
- **Modern UI**: Clean, responsive web interface
- **Debug Tools**: Face detection debugging and cache management

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- (Optional) GPU for faster processing with CNN model

### Installation

1. **Clone or download the project**:
```bash
git clone https://github.com/bhavyashah10/photo-finder.git
cd photo-finder
```

2. **Install required packages**:
```bash
pip install flask opencv-python face-recognition pillow numpy werkzeug
```

**Note**: Installing `face-recognition` can be challenging on some systems:
- **Windows**: May require Visual Studio Build Tools
- **macOS**: May need cmake: `brew install cmake`
- **Linux**: May need: `sudo apt-get install cmake libopenblas-dev liblapack-dev`

3. **Set up project structure**:
```
photo-finder/
├── app.py                    # Main Flask application
├── templates/                # HTML templates
│   ├── index.html
│   ├── photographer.html
│   └── guest.html
├── static/                   # CSS, JS, and uploads
│   ├── style.css
│   ├── script.js
│   └── uploads/
│       ├── wedding_photos/
│       └── guest_photos/
└── utils/
    └── face_matcher.py       # Face recognition logic
```

4. **Run the application**:
```bash
python3 app.py
```

5. **Open your browser** and go to: `http://localhost:5000`

## 📖 How to Use

### For Photographers

1. Navigate to **"I'm the Photographer"**
2. Upload all photos (supports multiple file selection)
3. Optionally run **pre-processing** for faster guest searches
4. View uploaded photos and system statistics

### For Guests

1. Navigate to **"I'm a Guest"**
2. Upload a clear photo of yourself
3. Wait for the AI to process and find matches
4. View and download all photos containing your face

## ⚙️ Configuration

### Face Recognition Settings

In `app.py`, you can adjust the face matching parameters:

```python
# Strict settings 
face_matcher = FaceMatcher(tolerance=0.45, model='cnn', min_confidence=0.6)

# Very strict (fewer false matches, may miss some real ones)
face_matcher = FaceMatcher(tolerance=0.4, model='cnn', min_confidence=0.7)

# More lenient (more matches, may include false positives)
face_matcher = FaceMatcher(tolerance=0.5, model='cnn', min_confidence=0.55)
```

### Parameters Explained

- **tolerance** (0.3-0.7): How strict face matching is. Lower = stricter
- **model** ('hog' or 'cnn'): Face detection model. CNN is more accurate but requires more processing power
- **min_confidence** (0.0-1.0): Minimum confidence score to accept a match

### Performance Optimization

- **Use 'hog' model** for CPU-only systems (faster but less accurate)
- **Use 'cnn' model** for GPU systems (slower but more accurate)
- **Pre-process photos** after upload for faster guest searches
- **Clear cache** if you change tolerance settings

## 🔧 API Endpoints

### Main Routes
- `GET /` - Homepage
- `GET /photographer` - Photographer portal
- `GET /guest` - Guest portal

### Upload & Processing
- `POST /upload_wedding_photos` - Upload photos
- `POST /find_matches` - Find matches for guest photo
- `POST /preprocess_photos` - Pre-process all photos

### Management
- `GET /get_wedding_photos` - List all photos
- `GET /get_cache_stats` - View cache statistics
- `POST /clear_cache` - Clear face encoding cache
- `GET /face_debug/<filename>` - Debug face detection for specific image

## 🎯 Best Practices

### For Best Results

**Guest Photos:**
- Use clear, well-lit photos
- Face should be clearly visible and front-facing
- One person per photo works best
- Avoid sunglasses or face coverings

**Event Photos:**
- High-resolution images work better
- Good lighting improves face detection
- Multiple angles of the same person help matching

### Troubleshooting

**"No face detected" error:**
- Ensure the photo has a clear, visible face
- Try a different photo with better lighting
- Check that the face isn't too small in the image

**Too many false matches:**
- Decrease tolerance (e.g., from 0.6 to 0.4)
- Increase min_confidence (e.g., from 0.6 to 0.7)
- Clear cache after changing settings

**Missing real matches:**
- Increase tolerance (e.g., from 0.4 to 0.5)
- Decrease min_confidence (e.g., from 0.7 to 0.6)
- Check photo quality and lighting

## 🛠️ Development

### File Structure Explained

- **app.py**: Main Flask application with routes and logic
- **face_matcher.py**: Core face recognition and matching algorithms
- **templates/**: HTML templates for the web interface
- **static/**: CSS, JavaScript, and uploaded images
- **uploads/**: Automatically created folders for photo storage

### Adding Features

The application is modular and easy to extend:

- Add new routes in `app.py`
- Modify face recognition logic in `utils/face_matcher.py`
- Update UI in HTML templates and CSS
- Add JavaScript functionality in `static/script.js`

### Database Integration

Currently uses file system storage. To add database support:

1. Replace file operations with database queries
2. Store face encodings in database instead of pickle cache
3. Add user authentication and photo ownership tracking

## 📊 Performance Notes

### Processing Times (approximate)

- **Face detection**: 1-3 seconds per photo (depends on size and model)
- **Guest search**: 0.1-0.5 seconds per cached photo
- **Initial processing**: 2-5 seconds per photo for cache building

### Memory Usage

- **Face encodings**: ~512 bytes per face
- **Image caching**: Minimal (only encodings are cached, not images)
- **Typical event**: 500 photos ≈ 1-2MB of cached encodings

## 🔒 Security Considerations

### For Production Use

- Add user authentication and authorization
- Implement file upload validation and sanitization
- Use HTTPS for secure photo transmission
- Consider privacy implications of face data storage
- Add rate limiting for API endpoints
- Validate and sanitize all user inputs

### Privacy Notes

- Guest photos are automatically deleted after processing
- Face encodings are mathematical representations, not actual images
- Consider implementing automatic data deletion policies
- Inform users about data processing and storage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is open source. Please check the LICENSE file for details.

## 🆘 Support

### Common Issues

1. **Installation problems**: Check the prerequisites and installation notes for your OS
2. **Performance issues**: Try using 'hog' model instead of 'cnn' for faster processing
3. **Accuracy problems**: Adjust tolerance and confidence settings
4. **Memory issues**: Clear cache periodically or process photos in smaller batches

### Getting Help

- Check the troubleshooting section above
- Review the configuration options
- Test with different tolerance settings
- Ensure photos meet the quality guidelines

## 🏗️ Future Enhancements

- [ ] Multiple face detection per guest photo
- [ ] Batch guest processing
- [ ] Advanced filtering and sorting options
- [ ] Mobile app version
- [ ] Cloud storage integration
- [ ] Real-time processing status
- [ ] Photo metadata preservation
- [ ] Automated photo organization
- [ ] Integration with popular photo sharing platforms

---

*Powered by Python, Flask, and face recognition technology*