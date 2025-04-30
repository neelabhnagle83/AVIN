import React, { useState } from 'react';
import { View, Text, TouchableOpacity, TextInput, Image, Alert, KeyboardAvoidingView, Platform, TouchableWithoutFeedback, Keyboard } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { styles } from '@/components/styles/LoginStyles';
import type { RootStackParamList } from '@/app/navigation/AppNavigator';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Login'>;

export default function LoginScreen() {
  const navigation = useNavigation<NavigationProp>();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = () => {
    if (!email.endsWith('@gmail.com')) {
      Alert.alert('Invalid Email', 'Please enter a valid Gmail address ending with @gmail.com');
      return;
    }

    const hasNumber = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);
    
    if (!hasNumber || !hasSpecialChar) {
      Alert.alert(
        'Weak Password', 
        'Password must contain at least:\n- 1 number\n- 1 special character'
      );
      return;
    }

    navigation.navigate('Dashboard');
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