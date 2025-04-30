import React from 'react';
import { View, Text, ImageBackground, TouchableOpacity } from 'react-native';
import { styles } from '@/components/styles/Splash2Styles';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/app/navigation/AppNavigator';

type Splash2ScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Splash2'>;

export default function Splash2Screen() {
  const navigation = useNavigation<Splash2ScreenNavigationProp>();

  return (
    <ImageBackground
      source={require('@/assets/images/splash2.png')}
      style={styles.background}
      resizeMode="cover"
    >
      <View style={styles.overlay} />

      <View style={styles.topContent}>
        <Text style={styles.heading}>Instant Solution</Text>
        <Text style={styles.subtext}>Scan or upload plant photos to detect issues and get instant solutions.</Text>
      </View>

      <TouchableOpacity style={styles.button} onPress={() => navigation.navigate('Splash3')}>
        <Text style={styles.buttonText}>Next</Text>
      </TouchableOpacity>
    </ImageBackground>
  );
}
