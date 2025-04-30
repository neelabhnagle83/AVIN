import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Alert, KeyboardAvoidingView, Platform, TouchableWithoutFeedback, Keyboard, ScrollView } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { styles } from '@/components/styles/PersonalInfo3Styles';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/app/navigation/AppNavigator';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'PersonalInfo3'>;

export default function PersonalInfo3Screen() {
  const navigation = useNavigation<NavigationProp>();
  const [selectedHelp, setSelectedHelp] = useState<string[]>([]);
  const [govtUpdates, setGovtUpdates] = useState('Not Selected');
  const [reminders, setReminders] = useState('Not Selected');
  const [openToNew, setOpenToNew] = useState('Not Selected');
  const [showGovtDropdown, setShowGovtDropdown] = useState(false);
  const [showRemindersDropdown, setShowRemindersDropdown] = useState(false);
  const [showOpenToNewDropdown, setShowOpenToNewDropdown] = useState(false);

  const handleBack = () => {
    navigation.goBack();
  };

  const handleFinish = () => {
    if (govtUpdates === 'Not Selected' || reminders === 'Not Selected' || openToNew === 'Not Selected') {
      Alert.alert('Missing Information', 'Please answer all questions');
      return;
    }

    if (selectedHelp.length === 0) {
      Alert.alert('Missing Information', 'Please select at least one help option');
      return;
    }

    navigation.navigate('Dashboard');
  };

  const toggleHelpOption = (option: string) => {
    if (selectedHelp.includes(option)) {
      setSelectedHelp(selectedHelp.filter(item => item !== option));
    } else {
      setSelectedHelp([...selectedHelp, option]);
    }
  };

  const helpOptions = [
    'Chat with Experts',
    'Smart Crop Suggestions',
    'Disease Detection',
    'Boost Income',
    'Learn Modern Farming',
    'Sell My Crops'
  ];

  return (
    <View style={{ flex: 1 }}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={{ flex: 1 }}
      >
        <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
          <ScrollView contentContainerStyle={{ flexGrow: 1 }}>
            <View style={styles.fullScreen}>
              <Text style={styles.header}>Profile setup</Text>
              
              {/* Progress Bar - Third step active */}
              <View style={styles.progressContainer}>
                <View style={[styles.progressStep, styles.progressInactive]} />
                <View style={[styles.progressStep, styles.progressInactive]} />
                <View style={[styles.progressStep, styles.progressActive]} />
              </View>

              <View style={styles.sectionContainer}>
                <Text style={styles.sectionTitle}>Goals & Support</Text>
                
                {/* Help Options Question */}
                <Text style={styles.questionText}>What Help you want from AVIN?</Text>
                <View style={styles.helpOptionsContainer}>
                  {helpOptions.map((option) => (
                    <TouchableOpacity
                      key={option}
                      style={[
                        styles.helpOption,
                        selectedHelp.includes(option) && styles.helpOptionSelected
                      ]}
                      onPress={() => toggleHelpOption(option)}
                    >
                      <Text style={[
                        styles.helpOptionText,
                        selectedHelp.includes(option) && styles.helpOptionTextSelected
                      ]}>
                        {option}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>

                {/* Government Updates Question */}
                <Text style={styles.questionText}>Are you interested in receiving government scheme updates?</Text>
                <View style={styles.inputContainer}>
                  <Ionicons name="megaphone" size={20} style={styles.icon} />
                  <TouchableOpacity 
                    style={styles.dropdownInput}
                    onPress={() => setShowGovtDropdown(!showGovtDropdown)}
                  >
                    <Text style={govtUpdates === 'Not Selected' ? styles.placeholderText : styles.selectedText}>
                      {govtUpdates}
                    </Text>
                    <Ionicons 
                      name={showGovtDropdown ? 'chevron-up' : 'chevron-down'} 
                      size={20} 
                      style={styles.dropdownIcon} 
                    />
                  </TouchableOpacity>
                  
                  {showGovtDropdown && (
                    <View style={styles.dropdownContainer}>
                      <TouchableOpacity 
                        style={styles.dropdownItem}
                        onPress={() => {
                          setGovtUpdates('Yes');
                          setShowGovtDropdown(false);
                        }}
                      >
                        <Text>Yes</Text>
                      </TouchableOpacity>
                      <TouchableOpacity 
                        style={styles.dropdownItem}
                        onPress={() => {
                          setGovtUpdates('No');
                          setShowGovtDropdown(false);
                        }}
                      >
                        <Text>No</Text>
                      </TouchableOpacity>
                    </View>
                  )}
                </View>

                {/* Reminders Question */}
                <Text style={styles.questionText}>Would you like reminders about pesticide usage, watering, or harvest time?</Text>
                <View style={styles.inputContainer}>
                  <Ionicons name="notifications" size={20} style={styles.icon} />
                  <TouchableOpacity 
                    style={styles.dropdownInput}
                    onPress={() => setShowRemindersDropdown(!showRemindersDropdown)}
                  >
                    <Text style={reminders === 'Not Selected' ? styles.placeholderText : styles.selectedText}>
                      {reminders}
                    </Text>
                    <Ionicons 
                      name={showRemindersDropdown ? 'chevron-up' : 'chevron-down'} 
                      size={20} 
                      style={styles.dropdownIcon} 
                    />
                  </TouchableOpacity>
                  
                  {showRemindersDropdown && (
                    <View style={styles.dropdownContainer}>
                      <TouchableOpacity 
                        style={styles.dropdownItem}
                        onPress={() => {
                          setReminders('Yes');
                          setShowRemindersDropdown(false);
                        }}
                      >
                        <Text>Yes</Text>
                      </TouchableOpacity>
                      <TouchableOpacity 
                        style={styles.dropdownItem}
                        onPress={() => {
                          setReminders('No');
                          setShowRemindersDropdown(false);
                        }}
                      >
                        <Text>No</Text>
                      </TouchableOpacity>
                    </View>
                  )}
                </View>

                {/* Open to New Techniques Question */}
                <Text style={styles.questionText}>Are you open to trying new crops or techniques if advised by AI?</Text>
                <View style={styles.inputContainer}>
                  <Ionicons name="bulb" size={20} style={styles.icon} />
                  <TouchableOpacity 
                    style={styles.dropdownInput}
                    onPress={() => setShowOpenToNewDropdown(!showOpenToNewDropdown)}
                  >
                    <Text style={openToNew === 'Not Selected' ? styles.placeholderText : styles.selectedText}>
                      {openToNew}
                    </Text>
                    <Ionicons 
                      name={showOpenToNewDropdown ? 'chevron-up' : 'chevron-down'} 
                      size={20} 
                      style={styles.dropdownIcon} 
                    />
                  </TouchableOpacity>
                  
                  {showOpenToNewDropdown && (
                    <View style={styles.dropdownContainer}>
                      <TouchableOpacity 
                        style={styles.dropdownItem}
                        onPress={() => {
                          setOpenToNew('Yes');
                          setShowOpenToNewDropdown(false);
                        }}
                      >
                        <Text>Yes</Text>
                      </TouchableOpacity>
                      <TouchableOpacity 
                        style={styles.dropdownItem}
                        onPress={() => {
                          setOpenToNew('No');
                          setShowOpenToNewDropdown(false);
                        }}
                      >
                        <Text>No</Text>
                      </TouchableOpacity>
                    </View>
                  )}
                </View>
              </View>

              {/* Dual Buttons */}
              <View style={styles.buttonContainer}>
                <TouchableOpacity 
                  style={[styles.button, styles.backButton]}
                  onPress={handleBack}
                >
                  <Text style={[styles.buttonText, styles.backButtonText]}>Back</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.button, styles.finishButton]}
                  onPress={handleFinish}
                >
                  <Text style={[styles.buttonText, styles.finishButtonText]}>Finish</Text>
                </TouchableOpacity>
              </View>
            </View>
          </ScrollView>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>
    </View>
  );
}