import React from 'react';
import { View, Text, ImageBackground, TouchableOpacity } from 'react-native';
import { styles } from '@/components/styles/Splash4Styles';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/app/navigation/AppNavigator';

type Splash4ScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Splash4'>;

export default function Splash4Screen() {
  const navigation = useNavigation<Splash4ScreenNavigationProp>();

  return (
    <ImageBackground
      source={require('@/assets/images/splash4.png')}
      style={styles.background}
      resizeMode="cover"
    >
      <View style={styles.overlay} />

      <View style={styles.topContent}>
        <Text style={styles.heading}>Your Fields, Our Guidance</Text>
        <Text style={styles.subtext}>Add your land, get tailored crop planning, and track progress.</Text>
      </View>

      <TouchableOpacity style={styles.button} onPress={() => navigation.navigate('ChooseLoginSignup')}>
        <Text style={styles.buttonText}>Let’s Get Started</Text>
      </TouchableOpacity>
    </ImageBackground>
  );
}
