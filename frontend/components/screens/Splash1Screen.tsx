import React from 'react';
import { View, Text, ImageBackground, TouchableOpacity } from 'react-native';
import { styles } from '@/components/styles/Splash1Styles';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/app/navigation/AppNavigator'; // You already defined this in AppNavigator

type Splash1ScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Splash1'>;

export default function Splash1Screen() {
  const navigation = useNavigation<Splash1ScreenNavigationProp>(); // ✅ use inside the component

  return (
    <ImageBackground
      source={require('@/assets/images/splash1.png')}
      style={styles.background}
      resizeMode="cover"
    >
      {/* Transparent black overlay */}
      <View style={styles.overlay} />

      {/* Content on top */}
      <View style={styles.topContent}>
        <Text style={styles.heading}>Welcome to AVIN</Text>
        <Text style={styles.subtext}>Your smart companion for farming and rural growth.</Text>
      </View>

      <TouchableOpacity style={styles.button} onPress={() => navigation.navigate('Splash2')}>
        <Text style={styles.buttonText}>Next</Text>
      </TouchableOpacity>
    </ImageBackground>
  );
}
