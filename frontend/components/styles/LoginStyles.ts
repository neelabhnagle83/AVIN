import { StyleSheet, Dimensions } from 'react-native';

const { width } = Dimensions.get('window');

export const styles = StyleSheet.create({
  fullScreen: {
    flex: 1,
    backgroundColor: '#fff',
  },
  logo: {
    width: 100,
    height: 100,
    alignSelf: 'center',
    marginTop: 64,
    marginBottom: -24,
    resizeMode: 'contain',
  },
  paragraph: {
    fontSize: 16,
    color: 'rgba(104, 107, 48, 0.85)',
    textAlign: 'center',
    marginTop: 0,
    marginBottom: 16,
    paddingHorizontal: 20,
  },
  leafImage: {
    position: 'absolute',
    top: '18%',
    right: 0,
    width: 100,
    height: 220,
    resizeMode: 'contain',
    zIndex: 1,
  },
  loginContainer: {
    flex: 1,
    width: '100%',
    backgroundColor: '#C7D3B1',
    borderTopLeftRadius: 25,
    borderTopRightRadius: 25,
    padding: 20,
    paddingTop: 20,
    marginTop: 80,
  },
  loginHeading: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#444702',
    marginBottom: 20,
    textAlign: 'left',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '100%',
    marginBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#686B30',
  },
  icon: {
    marginRight: 10,
    color: '#444702',
  },
  input: {
    width: '80%',
    height: 40,
    fontSize: 16,
    paddingHorizontal: 10,
  },
  forgotPassword: {
    alignSelf: 'flex-end',
    marginBottom: 16,
  },
  forgotPasswordText: {
    color: '#444702',
  },
  loginButton: {
    width: '100%',
    paddingVertical: 14,
    backgroundColor: 'transparent',
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 10,
    borderWidth: 1.5,
    borderColor: '#444702',
  },
  buttonText: {
    fontSize: 18,
    color: '#444702',
    fontWeight: 'bold',
  },
  orDivider: {
    marginVertical: 20,
    textAlign: 'center',
    fontSize: 16,
    color: '#444702',
  },
  socialButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: 20,
  },
  socialButton: {
    width: '30%',
    paddingVertical: 10,
    backgroundColor: 'transparent',
    borderRadius: 8,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: '#444702',
  },
  signUpText: {
    fontSize: 14,
    color: 'rgba(68, 71, 2, 0.9)',
    textAlign: 'center',
  },
  signUpLink: {
    color: '#444702',
    fontWeight: 'bold',
  },
  topDividerLine: {
    height: 4,
    backgroundColor: '#444702',
    width: '20%',
    alignSelf: 'center',
    marginBottom: 20,
    borderRadius: 2,
  },
});
