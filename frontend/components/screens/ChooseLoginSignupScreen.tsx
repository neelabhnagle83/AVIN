import React from 'react';
import { View, Text, ImageBackground, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/app/navigation/AppNavigator';
import { styles } from '@/components/styles/ChooseLoginSignupStyles';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'ChooseLoginSignup'>;

export default function ChooseLoginSignupScreen() {
  const navigation = useNavigation<NavigationProp>();

  return (
    <ImageBackground
      source={require('@/assets/images/choose.png')}
      style={styles.background}
      resizeMode="cover"
    >
      <View style={styles.overlay} />

      {/* Top Heading + Subtext */}
      <View style={styles.topContent}>
        <Text style={styles.heading}>Start Smart Farming</Text>
        <Text style={styles.subtext}>
          Get smart tips, manage your land, and grow better with AVIN. Sign in to begin.
        </Text>
      </View>

      {/* Bottom Buttons */}
      <View style={styles.bottomContent}>
        <TouchableOpacity style={styles.button} onPress={() => navigation.navigate('Login')}>
          <Text style={styles.buttonText}>Login</Text>
        </TouchableOpacity>

        <TouchableOpacity onPress={() => navigation.navigate('Signup')}>
          <Text style={styles.linkText}>
            Don’t have an account? <Text style={styles.link}>Sign Up</Text>
          </Text>
        </TouchableOpacity>
      </View>
    </ImageBackground>
  );
}
