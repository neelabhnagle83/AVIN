import React, { useState } from 'react';
import { View, Text, TouchableOpacity, TextInput, Image, Alert, KeyboardAvoidingView, Platform, TouchableWithoutFeedback, Keyboard } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { styles } from '@/components/styles/LoginStyles';
import type { RootStackParamList } from '@/app/navigation/AppNavigator';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import axios from 'axios';
import { API_BASE_URL } from '@/constants/Config';
type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Login'>;

export default function LoginScreen() {
  const navigation = useNavigation<NavigationProp>();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Missing Information', 'Please fill in all fields');
      return;
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/auth/login`, {
        email,
        password,
      });

      if (response.status === 200) {
        Alert.alert('Success', 'Login successful!');
        navigation.navigate('Dashboard');
      } else {
        Alert.alert('Error', 'Invalid email or password.');
      }
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        console.error('Backend error during login:', error.response.data);
        Alert.alert('Error', error.response.data.message || 'Invalid email or password.');
      } else {
        console.error('Unexpected error during login:', error);
        Alert.alert('Error', 'An unexpected error occurred during login.');
      }
    }
  };

  return (
    <View style={{ flex: 1 }}>
      {/* Fixed Leaf Image - now truly fixed */}
      <Image
            source={require('@/assets/images/login_leaf.png')}
            style={[styles.leafImage, {
              position: 'absolute',
              top: 140,
              right: 0,
              zIndex: 2,
            }]}
          />

      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={{ flex: 1 }}
      >
        <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
          <View style={styles.fullScreen}>
            {/* Logo */}
            <Image
              source={require('@/assets/images/logo.png')}
              style={styles.logo}
            />

            {/* Paragraph */}
            <Text style={styles.paragraph}>
              Welcome to AVIN, your smart farming companion. Please login to continue.
            </Text>

            {/* Bottom Container */}
            <View style={styles.loginContainer}>
              {/* Top Divider */}
              <View style={styles.topDividerLine} />

              <Text style={styles.loginHeading}>Login</Text>

              {/* Email Input */}
              <View style={styles.inputContainer}>
                <Ionicons name="mail" size={20} style={styles.icon} />
                <TextInput
                  placeholder="Email"
                  style={styles.input}
                  keyboardType="email-address"
                  value={email}
                  onChangeText={setEmail}
                />
              </View>

              {/* Password Input */}
              <View style={styles.inputContainer}>
                <Ionicons name="lock-closed" size={20} style={styles.icon} />
                <TextInput
                  placeholder="Password"
                  style={styles.input}
                  secureTextEntry
                  value={password}
                  onChangeText={setPassword}
                />
              </View>

              {/* Forgot Password */}
              <TouchableOpacity style={styles.forgotPassword}>
                <Text style={styles.forgotPasswordText}>Forgot Password?</Text>
              </TouchableOpacity>

              {/* Login Button */}
              <TouchableOpacity 
                style={styles.loginButton} 
                onPress={handleLogin}
              >
                <Text style={styles.buttonText}>Login</Text>
              </TouchableOpacity>

              {/* Divider */}
              <Text style={styles.orDivider}>----------or----------</Text>

              {/* Social Login */}
              <View style={styles.socialButtons}>
                <TouchableOpacity 
                  style={styles.socialButton} 
                  onPress={() => navigation.navigate('PhoneLogin')}
                >
                  <Ionicons name="call" size={20} color="#686B30" />
                </TouchableOpacity>
                <TouchableOpacity style={styles.socialButton}>
                  <Ionicons name="logo-google" size={20} color="#686B30" />
                </TouchableOpacity>
                <TouchableOpacity style={styles.socialButton}>
                  <Ionicons name="logo-facebook" size={20} color="#686B30" />
                </TouchableOpacity>
              </View>

              {/* Sign Up Link */}
              <TouchableOpacity onPress={() => navigation.navigate('Signup')}>
                <Text style={styles.signUpText}>
                  Don't have an account? <Text style={styles.signUpLink}>Sign Up</Text>
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>
    </View>
  );
}