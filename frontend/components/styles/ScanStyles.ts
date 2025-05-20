import { StyleSheet } from 'react-native';

export default StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 16,
    paddingBottom: 60,
  },
  heading: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#444702',
    marginBottom: 4,
    marginTop: 48,
  },
  subtext: {
    fontSize: 14,
    color: '#444',
    marginBottom: 16,
    textAlign: 'left',
  },
  buttonRow: {
    flexDirection: 'row',
    marginBottom: 16,
  },
  uploadButton: {
    flex: 1,
    backgroundColor: '#444702',
    padding: 10,
    borderRadius: 6,
    marginRight: 8,
    height: 50,
    justifyContent: 'center',
    alignItems: 'center',
  },
  uploadText: {
    color: '#fff',
    fontWeight: '600',
  },
  takePhotoButton: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#444702',
    padding: 10,
    borderRadius: 6,
    height: 50,
    justifyContent: 'center',
    alignItems: 'center',
  },
  takePhotoText: {
    color: '#444702',
    fontWeight: '600',
  },
  previewBox: {
    backgroundColor: '#ecffbb',
    borderRadius: 8,
    padding: 10,
    marginBottom: 16,
  },
  photoInfo: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 6,
  },
  maxSize: {
    color: 'red',
    fontSize: 12,
  },
  imageList: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  imageWrapper: {
    position: 'relative',
    marginRight: 8,
  },
  imageThumb: {
    width: 50,
    height: 50,
    borderRadius: 4,
  },
  removeIcon: {
    position: 'absolute',
    top: -6,
    right: -6,
    backgroundColor: '#fff',
    borderRadius: 10,
  },
  addMoreBtn: {
    width: 50,
    height: 50,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: '#444702',
    justifyContent: 'center',
    alignItems: 'center',
  },
  scanBox: {
    backgroundColor: '#ecffbb',
    borderRadius: 8,
    padding: 20,
    alignItems: 'center',
  },
  waitingText: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 16,
  },
  scanButton: {
    backgroundColor: '#dfffb8',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#333',
    paddingVertical: 12,
    paddingHorizontal: 32,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  scanButtonText: {
    color: '#000',
    fontWeight: 'bold',
  },
  resultText: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#444702',
    marginTop: 10
  },
  confidenceText: {
    fontSize: 16,
    marginBottom: 15,
    color: '#686B30',
    fontStyle: 'italic'
  },
});
