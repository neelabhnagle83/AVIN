import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import Splash1Screen from '@/components/screens/Splash1Screen';
import Splash2Screen from '@/components/screens/Splash2Screen';
import Splash3Screen from '@/components/screens/Splash3Screen';
import Splash4Screen from '@/components/screens/Splash4Screen';
import ChooseLoginSignupScreen from '@/components/screens/ChooseLoginSignupScreen';
import LoginScreen from '@/components/screens/LoginScreen';
import SignupScreen from '@/components/screens/SignupScreen';
import PersonalInfo1Screen from '@/components/screens/PersonalInfo1Screen';
import PersonalInfo2Screen from '@/components/screens/PersonalInfo2Screen';
import PersonalInfo3Screen from '@/components/screens/PersonalInfo3Screen';
import DashboardScreen from '@/components/screens/DashboardScreen';
import PhoneLoginScreen from '@/components/screens/PhoneLogin';

export type RootStackParamList = {
  Splash1: undefined;
  Splash2: undefined;
  Splash3: undefined;
  Splash4: undefined;
  ChooseLoginSignup: undefined;
  Login: undefined;
  Signup: undefined;
  PersonalInfo1: undefined;
  PersonalInfo2: undefined;
  PersonalInfo3: undefined;
  Dashboard: undefined;
  PhoneLogin: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function AppNavigator() {
  return (
    <Stack.Navigator 
      initialRouteName="Splash1"
      screenOptions={{
        headerShown: false,
        animation: 'slide_from_right',
      }}
    >
      <Stack.Screen name="Splash1" component={Splash1Screen} />
      <Stack.Screen name="Splash2" component={Splash2Screen} />
      <Stack.Screen name="Splash3" component={Splash3Screen} />
      <Stack.Screen name="Splash4" component={Splash4Screen} />
      <Stack.Screen name="ChooseLoginSignup" component={ChooseLoginSignupScreen} />
      <Stack.Screen name="Login" component={LoginScreen} />
      <Stack.Screen name="Signup" component={SignupScreen} />
      <Stack.Screen name="PersonalInfo1" component={PersonalInfo1Screen} />
      <Stack.Screen name="PersonalInfo2" component={PersonalInfo2Screen} />
      <Stack.Screen name="PersonalInfo3" component={PersonalInfo3Screen} />
      <Stack.Screen name="Dashboard" component={DashboardScreen} />
      <Stack.Screen name="PhoneLogin" component={PhoneLoginScreen} />
    </Stack.Navigator>
  );
}