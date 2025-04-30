import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
    background: {
      flex: 1,
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    overlay: {
        position: 'absolute',
        top: 0,
        right: 0,
        bottom: 0,
        left: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.4)',
      },
    topContent: {
      width: '100%',
      paddingTop: 80,
      paddingHorizontal: 20,
      alignItems: 'center',
    },
    heading: {
      fontSize: 28,
      fontWeight: 'bold',
      color: '#fff',
      textAlign: 'center',
      fontFamily: 'Roboto',
      marginBottom: 8,
    },
    subtext: {
      fontSize: 16,
      color: '#fff',
      textAlign: 'center',
      fontFamily: 'Roboto',
    },
    bottomContent: {
      width: '100%',
      padding: 20,
      alignItems: 'center',
    },
    button: {
      width: '100%',
      paddingVertical: 14,
      backgroundColor: '#F5DC5C',
      borderRadius: 8,
      alignItems: 'center',
      marginBottom: 16,
    },
    buttonText: {
      fontSize: 18,
      color: '#000',
      fontWeight: 'bold',
      fontFamily: 'Roboto',
    },
    linkText: {
      color: '#fff',
      fontSize: 14,
      fontFamily: 'Roboto',
    },
    link: {
      fontWeight: 'bold',
      textDecorationLine: 'underline',
    },
  });
  