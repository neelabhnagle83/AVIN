import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Image,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Keyboard,
  TouchableWithoutFeedback,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Checkbox from 'expo-checkbox';
import { styles } from '@/components/styles/LoginStyles';
import CountryPicker from './PhoneLoginPicker';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/app/navigation/AppNavigator';

type PhoneLoginNavigationProp = NativeStackNavigationProp<RootStackParamList, 'PhoneLogin'>;

export default function PhoneLogin() {
  const navigation = useNavigation<PhoneLoginNavigationProp>();
  const [countryCode, setCountryCode] = useState('+91');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [agree, setAgree] = useState(false);
  const [showOtpScreen, setShowOtpScreen] = useState(false);
  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const otpInputRefs = Array(6).fill(0).map(() => React.createRef<TextInput>());

  const handleSendOtp = () => {
    if (phoneNumber && agree) {
      setShowOtpScreen(true);
    }
  };

  const handleOtpChange = (index: number, value: string) => {
    if (value.length > 1) {
      value = value[value.length - 1];
    }

    const newOtp = [...otp];
    newOtp[index] = value;
    setOtp(newOtp);

    if (value && index < 5) {
      otpInputRefs[index + 1].current?.focus();
    }

    if (newOtp.every((digit) => digit !== '')) {
      Keyboard.dismiss();
    }
  };

  const handleOtpKeyPress = (index: number, key: string) => {
    if (key === 'Backspace' && !otp[index] && index > 0) {
      otpInputRefs[index - 1].current?.focus();
    }
  };

  const handleResendOtp = () => {
    setOtp(['', '', '', '', '', '']);
    otpInputRefs[0].current?.focus();
  };

  const handleVerifySubmit = () => {
    // Check if all OTP digits are filled
    if (otp.every(digit => digit !== '')) {
      navigation.navigate('Dashboard');
    }
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      style={{ flex: 1 }}
    >
      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <View style={{ flex: 1 }}>
          {/* Leaf stays fixed */}
          <Image
            source={require('@/assets/images/login_leaf.png')}
            style={[styles.leafImage, {
              position: 'absolute',
              top: 140,
              right: 0,
              zIndex: 2,
            }]}
          />
          <ScrollView contentContainerStyle={{ flexGrow: 1 }} keyboardShouldPersistTaps="handled">
            <View style={styles.fullScreen}>
              {/* Logo */}
              <Image source={require('@/assets/images/logo.png')} style={styles.logo} />
  
              {/* Paragraph top */}
              <Text style={[styles.paragraph, {
                alignSelf: 'flex-start',
                marginLeft: 20,
                marginRight: 20,
              }]}>
                Welcome to AVIN, your smart farming companion. Please login to continue.
              </Text>
  
              {/* Bottom container */}
              <View style={styles.loginContainer}>
                <View style={styles.topDividerLine} />
                <Text style={[styles.loginHeading, { alignSelf: 'flex-start', marginBottom: 5 }]}>Login</Text>
  
                {!showOtpScreen ? (
                  <>
                    {/* Phone Input Screen */}
                    <View style={{ alignSelf: 'flex-start', marginBottom: 5, marginLeft: -20 }}>
                      <Text style={[styles.paragraph]}>
                        We'll send you a 6-digit OTP to verify.
                      </Text>
                    </View>
  
                    <View style={styles.inputContainer}>
                      <CountryPicker selectedCode={countryCode} onSelectCode={setCountryCode} />
                      <TextInput
                        placeholder="Phone Number"
                        keyboardType="phone-pad"
                        value={phoneNumber}
                        onChangeText={setPhoneNumber}
                        style={[styles.input, { flex: 1, color: '#444702' }]}
                      />
                    </View>
  
                    <View style={{
                      flexDirection: 'row',
                      alignItems: 'flex-start',
                      marginVertical: 12,
                      gap: 8,
                      marginLeft: 0,
                      marginRight: 20,
                      alignSelf: 'flex-start',
                    }}>
                      <Checkbox value={agree} onValueChange={setAgree} color={agree ? '#444702' : undefined} />
                      <Text style={{ color: '#444702', flex: 1 }}>
                        By continuing, you agree to our{' '}
                        <Text style={{ textDecorationLine: 'underline' }}>Terms & Privacy Policy</Text>
                      </Text>
                    </View>
  
                    <TouchableOpacity 
                      style={styles.loginButton}
                      onPress={handleSendOtp}
                    >
                      <Text style={styles.buttonText}>Send OTP</Text>
                    </TouchableOpacity>
                  </>
                ) : (
                  <>
                    {/* OTP Verification Screen */}
                    <View style={{ 
                      alignSelf: 'flex-start', 
                      marginBottom: 20,
                      marginLeft: -20,
                    }}>
                      <Text style={[styles.paragraph, { textAlign: 'left' }]}>
                        Enter the 6 digit code sent to {countryCode} {phoneNumber}
                      </Text>
                    </View>
  
                    <View style={{
                      flexDirection: 'row',
                      justifyContent: 'space-between',
                      marginBottom: 10,
                      width: '100%',
                      paddingHorizontal: 20,
                    }}>
                      {otp.map((digit, index) => (
                        <TextInput
                          key={index}
                          ref={otpInputRefs[index]}
                          style={{
                            width: 40,
                            height: 50,
                            borderWidth: 1,
                            borderColor: '#ccc',
                            borderRadius: 8,
                            textAlign: 'center',
                            fontSize: 18,
                            color: '#444702',
                            backgroundColor: 'white',
                          }}
                          keyboardType="number-pad"
                          maxLength={1}
                          value={digit}
                          onChangeText={(value) => handleOtpChange(index, value)}
                          onKeyPress={({ nativeEvent: { key } }) => handleOtpKeyPress(index, key)}
                        />
                      ))}
                    </View>
  
                    <TouchableOpacity 
                      style={styles.loginButton}
                      onPress={handleVerifySubmit}
                    >
                      <Text style={styles.buttonText}>Verify & Submit</Text>
                    </TouchableOpacity>
  
                    <View style={{ marginTop: 20, alignSelf: 'center' }}>
                      <Text style={{ color: '#444702' }}>
                        Didn't receive OTP?{' '}
                        <Text 
                          style={{ fontWeight: 'bold', textDecorationLine: 'underline' }}
                          onPress={handleResendOtp}
                        >
                          Resend
                        </Text>
                      </Text>
                    </View>
                  </>
                )}
              </View>
  
              {/* Back Button - Centered at Bottom */}
              <TouchableOpacity
                onPress={() => showOtpScreen ? setShowOtpScreen(false) : navigation.goBack()}
                style={{
                  position: 'absolute',
                  bottom: 20,
                  alignSelf: 'center',
                  zIndex: 10,
                }}
                hitSlop={{ top: 20, bottom: 20, left: 20, right: 20 }}
              >
                <Text style={{ 
                  color: '#444702', 
                  fontWeight: '600',
                }}>{'<-- Back'}</Text>
              </TouchableOpacity>
            </View>
          </ScrollView>
        </View>
      </TouchableWithoutFeedback>
    </KeyboardAvoidingView>
  );
}