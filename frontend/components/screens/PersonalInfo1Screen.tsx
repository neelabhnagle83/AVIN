import React, { useState } from 'react';
import { View, Text, TouchableOpacity, TextInput, Alert, KeyboardAvoidingView, Platform, TouchableWithoutFeedback, Keyboard } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { styles } from '@/components/styles/PersonalInfo1Styles';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/app/navigation/AppNavigator';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'PersonalInfo1'>;

export default function PersonalInfo1Screen() {
  const navigation = useNavigation<NavigationProp>();
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('Not Selected');
  const [district, setDistrict] = useState('');
  const [showGenderDropdown, setShowGenderDropdown] = useState(false);

  const handleNext = () => {
    if (!name || !age || gender === 'Not Selected' || !district) {
      Alert.alert('Missing Information', 'Please fill in all fields');
      return;
    }

    if (isNaN(Number(age))) {
      Alert.alert('Invalid Age', 'Please enter a valid number for age');
      return;
    }

    navigation.navigate('PersonalInfo2');
  };

  const handleGenderSelect = (selectedGender: string) => {
    setGender(selectedGender);
    setShowGenderDropdown(false);
  };

  return (
    <View style={{ flex: 1 }}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={{ flex: 1 }}
      >
        <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
          <View style={styles.fullScreen}>
            <Text style={styles.header}>Profile setup</Text>
            
            {/* Progress Bar */}
            <View style={styles.progressContainer}>
              <View style={[styles.progressStep, styles.progressActive]} />
              <View style={[styles.progressStep, styles.progressInactive]} />
              <View style={[styles.progressStep, styles.progressInactive]} />
            </View>

            <View style={styles.sectionContainer}>
              <Text style={styles.sectionTitle}>Personal Information</Text>
              
              <View style={styles.inputContainer}>
                <Ionicons name="person" size={20} style={styles.icon} />
                <TextInput
                  placeholder="Your Name"
                  style={styles.input}
                  value={name}
                  onChangeText={setName}
                />
              </View>

              <View style={styles.inputContainer}>
                <Ionicons name="calendar" size={20} style={styles.icon} />
                <TextInput
                  placeholder="Your Age"
                  style={styles.input}
                  keyboardType="numeric"
                  value={age}
                  onChangeText={setAge}
                />
              </View>

              <View style={styles.inputContainer}>
                <Ionicons name="transgender" size={20} style={styles.icon} />
                <TouchableOpacity 
                  style={styles.genderInput}
                  onPress={() => setShowGenderDropdown(!showGenderDropdown)}
                >
                  <Text style={gender === 'Not Selected' ? styles.placeholderText : styles.selectedText}>
                    {gender}
                  </Text>
                  <Ionicons 
                    name={showGenderDropdown ? 'chevron-up' : 'chevron-down'} 
                    size={20} 
                    style={styles.dropdownIcon} 
                  />
                </TouchableOpacity>
                
                {showGenderDropdown && (
                  <View style={styles.dropdownContainer}>
                    <TouchableOpacity 
                      style={styles.dropdownItem}
                      onPress={() => handleGenderSelect('Male')}
                    >
                      <Text>Male</Text>
                    </TouchableOpacity>
                    <TouchableOpacity 
                      style={styles.dropdownItem}
                      onPress={() => handleGenderSelect('Female')}
                    >
                      <Text>Female</Text>
                    </TouchableOpacity>
                    <TouchableOpacity 
                      style={styles.dropdownItem}
                      onPress={() => handleGenderSelect('Other')}
                    >
                      <Text>Other</Text>
                    </TouchableOpacity>
                  </View>
                )}
              </View>

              <View style={styles.inputContainer}>
                <Ionicons name="location" size={20} style={styles.icon} />
                <TextInput
                  placeholder="District or State"
                  style={styles.input}
                  value={district}
                  onChangeText={setDistrict}
                />
              </View>
            </View>

            <TouchableOpacity 
              style={styles.nextButton}
              onPress={handleNext}
            >
              <Text style={styles.buttonText}>Next</Text>
            </TouchableOpacity>
          </View>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>
    </View>
  );
}