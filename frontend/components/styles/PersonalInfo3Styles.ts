import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
  fullScreen: {
    flex: 1,
    padding: 20,
    backgroundColor: '#fff',
  },
  header: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    marginTop: 56,
    color: '#333',
    textAlign: 'center',
  },
  progressContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 30,
  },
  progressStep: {
    width: 30,
    height: 6,
    borderRadius: 3,
    marginHorizontal: 4,
  },
  progressActive: {
    backgroundColor: '#686B30',
  },
  progressInactive: {
    backgroundColor: '#E0E0E0',
  },
  sectionContainer: {
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#333',
  },
  questionText: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 10,
    color: '#333',
  },
  helpOptionsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 20,
  },
  helpOption: {
    borderWidth: 1,
    borderColor: '#686B30',
    borderRadius: 20,
    paddingVertical: 8,
    paddingHorizontal: 12,
    marginRight: 8,
    marginBottom: 8,
  },
  helpOptionSelected: {
    backgroundColor: '#686B30',
  },
  helpOptionText: {
    color: '#686B30',
    fontSize: 14,
  },
  helpOptionTextSelected: {
    color: '#fff',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    position: 'relative',
  },
  icon: {
    marginRight: 10,
    color: '#686B30',
  },
  dropdownInput: {
    flex: 1,
    borderBottomWidth: 1,
    borderBottomColor: '#ccc',
    paddingVertical: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  placeholderText: {
    color: '#999',
  },
  selectedText: {
    color: '#000',
  },
  dropdownIcon: {
    color: '#686B30',
  },
  dropdownContainer: {
    position: 'absolute',
    top: 50,
    left: 30,
    right: 0,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
    zIndex: 10,
  },
  dropdownItem: {
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 20,
  },
  button: {
    flex: 1,
    padding: 15,
    borderRadius: 5,
    alignItems: 'center',
  },
  backButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#686B30',
    marginRight: 10,
  },
  finishButton: {
    backgroundColor: '#686B30',
    marginLeft: 10,
  },
  buttonText: {
    fontWeight: 'bold',
    fontSize: 16,
  },
  backButtonText: {
    color: '#444702',
  },
  finishButtonText: {
    color: '#fff',
  },
});