import React, { useState } from 'react';
import { View, Text, TouchableOpacity, TextInput, Image, Alert, KeyboardAvoidingView, Platform, TouchableWithoutFeedback, Keyboard } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { styles } from '@/components/styles/SignupStyles';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/app/navigation/AppNavigator';
import CountryPicker from './PhoneLoginPicker';
import axios from 'axios';
import { API_BASE_URL } from '@/constants/Config';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Signup'>;

export default function SignupScreen() {
  const navigation = useNavigation<NavigationProp>();
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [countryCode, setCountryCode] = useState('+91');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [password, setPassword] = useState('');

  const handleSignup = async () => {
    // Validate all fields
    if (!email || !name || !phoneNumber || !password) {
      Alert.alert('Missing Information', 'Please fill in all fields');
      return;
    }

    if (!email.endsWith('@gmail.com')) {
      Alert.alert('Invalid Email', 'Please use a Gmail address ending with @gmail.com');
      return;
    }

    const hasNumber = /\d/.test(password);
    // Removed special character validation
    if (!hasNumber) {
      Alert.alert(
        'Weak Password', 
        'Password must contain:\n- At least 1 number'
      );
      return;
    }

    console.log('Password being sent:', password);

    try {
      const response = await axios.post(`${API_BASE_URL}/auth/register`, {
        email,
        name,
        phone: phoneNumber, // Send only the 10-digit phone number without the country code
        password,
      });

      if (response.status === 201) {
        Alert.alert('Success', 'User registered successfully!');
        navigation.navigate('PersonalInfo1');
      } else {
        Alert.alert('Error', 'Failed to register user.');
      }
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        if (error.response) {
          console.error('Backend error during signup:', error.response.data);
          Alert.alert('Error', error.response.data.message || 'An error occurred during signup.');
        } else {
          console.error('Axios error during signup:', error.message);
          Alert.alert('Network Error', 'Unable to connect to the server. Please check your network or try again later.');
        }
      } else {
        console.error('Unexpected error during signup:', error);
        Alert.alert('Error', 'An unexpected error occurred during signup.');
      }
    }
  };

  return (
    <View style={{ flex: 1 }}>
      {/* Fixed Leaf Image on left */}
      <Image
        source={require('@/assets/images/signup_leaf.png')}
        style={styles.signupLeafImage}
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

            {/* Welcome Text */}
            <Text style={styles.paragraph}>
              Helps you begin your digital farming journey
            </Text>

            {/* Signup Form Container */}
            <View style={styles.signupContainer}>
              <View style={styles.topDividerLine} />
              <Text style={styles.signupHeading}>Sign Up</Text>

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

              {/* Name Input */}
              <View style={styles.inputContainer}>
                <Ionicons name="person" size={20} style={styles.icon} />
                <TextInput
                  placeholder="Full Name"
                  style={styles.input}
                  value={name}
                  onChangeText={setName}
                />
              </View>

              {/* Phone Input */}
              <View style={styles.phoneInputContainer}>
                <CountryPicker selectedCode={countryCode} onSelectCode={setCountryCode} />
                <TextInput
                  placeholder="Phone Number"
                  style={[styles.input, { marginLeft: 10 }]}
                  keyboardType="phone-pad"
                  value={phoneNumber}
                  onChangeText={setPhoneNumber}
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

              {/* Sign Up Button */}
              <TouchableOpacity 
                style={styles.signupButton}
                onPress={handleSignup}
              >
                <Text style={styles.buttonText}>Sign Up</Text>
              </TouchableOpacity>

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

              {/* Login Link */}
              <TouchableOpacity onPress={() => navigation.navigate('Login')}>
                <Text style={styles.loginText}>
                  Already have an account? <Text style={styles.loginLink}>Login</Text>
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>
    </View>
  );
}