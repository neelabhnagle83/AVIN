import React from 'react';
import { View, Text, ImageBackground, TouchableOpacity } from 'react-native';
import { styles } from '@/components/styles/Splash3Styles';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/app/navigation/AppNavigator';

type Splash3ScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Splash3'>;

export default function Splash3Screen() {
  const navigation = useNavigation<Splash3ScreenNavigationProp>();

  return (
    <ImageBackground
      source={require('@/assets/images/splash3.png')}
      style={styles.background}
      resizeMode="cover"
    >
      <View style={styles.overlay} />

      <View style={styles.bottomContent}>
        <View style={styles.topContent}>
          <Text style={styles.heading}>Ask, Learn & Grow</Text>
          <Text style={styles.subtext}>Use our AI to get answers for farming tips, land suggestions, and more.</Text>
        </View>

        <TouchableOpacity style={styles.button} onPress={() => navigation.navigate('Splash4')}>
          <Text style={styles.buttonText}>Next</Text>
        </TouchableOpacity>
      </View>
    </ImageBackground>
  );
}
