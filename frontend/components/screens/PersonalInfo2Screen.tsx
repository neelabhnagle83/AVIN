import React, { useState } from 'react';
import { View, Text, TouchableOpacity, TextInput, Alert, KeyboardAvoidingView, Platform, TouchableWithoutFeedback, Keyboard, ScrollView } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { styles } from '@/components/styles/PersonalInfo2Styles';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/app/navigation/AppNavigator';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'PersonalInfo2'>;

export default function PersonalInfo2Screen() {
  const navigation = useNavigation<NavigationProp>();
  const [isFullTimeFarmer, setIsFullTimeFarmer] = useState('Not Selected');
  const [experience, setExperience] = useState('Not Selected');
  const [farmingTypes, setFarmingTypes] = useState<string[]>([]);
  const [sellsInLocalMarket, setSellsInLocalMarket] = useState('Not Selected');
  const [showFarmerDropdown, setShowFarmerDropdown] = useState(false);
  const [showExperienceDropdown, setShowExperienceDropdown] = useState(false);
  const [showMarketDropdown, setShowMarketDropdown] = useState(false);

  const farmingOptions = [
    'Crop Farming',
    'Dairy Farming',
    'Poultry Farming',
    'Organic Farming',
    'Aquaculture',
    'Horticulture',
    'Floriculture',
    'Contract Farming'
  ];

  const handleBack = () => {
    navigation.goBack();
  };

  const handleNext = () => {
    if (isFullTimeFarmer === 'Not Selected' || experience === 'Not Selected' || sellsInLocalMarket === 'Not Selected') {
      Alert.alert('Missing Information', 'Please answer all questions');
      return;
    }

    if (farmingTypes.length === 0) {
      Alert.alert('Missing Information', 'Please select at least one farming type');
      return;
    }

    navigation.navigate('PersonalInfo3');
  };

  const toggleFarmingType = (type: string) => {
    if (farmingTypes.includes(type)) {
      setFarmingTypes(farmingTypes.filter(t => t !== type));
    } else {
      setFarmingTypes([...farmingTypes, type]);
    }
  };

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
              
              {/* Progress Bar - Second step active */}
              <View style={styles.progressContainer}>
                <View style={[styles.progressStep, styles.progressInactive]} />
                <View style={[styles.progressStep, styles.progressActive]} />
                <View style={[styles.progressStep, styles.progressInactive]} />
              </View>

              <View style={styles.sectionContainer}>
                <Text style={styles.sectionTitle}>Farming Background</Text>
                
                {/* Full Time Farmer Question */}
                <Text style={styles.questionText}>Are you a full time Farmer?</Text>
                <View style={styles.inputContainer}>
                  <Ionicons name="person" size={20} style={styles.icon} />
                  <TouchableOpacity 
                    style={styles.dropdownInput}
                    onPress={() => setShowFarmerDropdown(!showFarmerDropdown)}
                  >
                    <Text style={isFullTimeFarmer === 'Not Selected' ? styles.placeholderText : styles.selectedText}>
                      {isFullTimeFarmer}
                    </Text>
                    <Ionicons 
                      name={showFarmerDropdown ? 'chevron-up' : 'chevron-down'} 
                      size={20} 
                      style={styles.dropdownIcon} 
                    />
                  </TouchableOpacity>
                  
                  {showFarmerDropdown && (
                    <View style={styles.dropdownContainer}>
                      <TouchableOpacity 
                        style={styles.dropdownItem}
                        onPress={() => {
                          setIsFullTimeFarmer('Yes');
                          setShowFarmerDropdown(false);
                        }}
                      >
                        <Text>Yes</Text>
                      </TouchableOpacity>
                      <TouchableOpacity 
                        style={styles.dropdownItem}
                        onPress={() => {
                          setIsFullTimeFarmer('No');
                          setShowFarmerDropdown(false);
                        }}
                      >
                        <Text>No</Text>
                      </TouchableOpacity>
                    </View>
                  )}
                </View>

                {/* Farming Experience Question */}
                <Text style={styles.questionText}>Years of Farming Experience:</Text>
                <View style={styles.inputContainer}>
                  <Ionicons name="calendar" size={20} style={styles.icon} />
                  <TouchableOpacity 
                    style={styles.dropdownInput}
                    onPress={() => setShowExperienceDropdown(!showExperienceDropdown)}
                  >
                    <Text style={experience === 'Not Selected' ? styles.placeholderText : styles.selectedText}>
                      {experience}
                    </Text>
                    <Ionicons 
                      name={showExperienceDropdown ? 'chevron-up' : 'chevron-down'} 
                      size={20} 
                      style={styles.dropdownIcon} 
                    />
                  </TouchableOpacity>
                  
                  {showExperienceDropdown && (
                    <View style={styles.dropdownContainer}>
                      {['No Experience', '0-1 years', '1-3 years', '3-5 years', 'More than 5 years'].map((item) => (
                        <TouchableOpacity 
                          key={item}
                          style={styles.dropdownItem}
                          onPress={() => {
                            setExperience(item);
                            setShowExperienceDropdown(false);
                          }}
                        >
                          <Text>{item}</Text>
                        </TouchableOpacity>
                      ))}
                    </View>
                  )}
                </View>

                {/* Farming Types Question - Updated to match PersonalInfo3 */}
                <Text style={styles.questionText}>What types of farming do you do?</Text>
                <View style={styles.helpOptionsContainer}>
                  {farmingOptions.map((option) => (
                    <TouchableOpacity
                      key={option}
                      style={[
                        styles.helpOption,
                        farmingTypes.includes(option) && styles.helpOptionSelected
                      ]}
                      onPress={() => toggleFarmingType(option)}
                    >
                      <Text style={[
                        styles.helpOptionText,
                        farmingTypes.includes(option) && styles.helpOptionTextSelected
                      ]}>
                        {option}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>

                {/* Local Market Question */}
                <Text style={styles.questionText}>Do you sell your produce in local markets?</Text>
                <View style={styles.inputContainer}>
                  <Ionicons name="cart" size={20} style={styles.icon} />
                  <TouchableOpacity 
                    style={styles.dropdownInput}
                    onPress={() => setShowMarketDropdown(!showMarketDropdown)}
                  >
                    <Text style={sellsInLocalMarket === 'Not Selected' ? styles.placeholderText : styles.selectedText}>
                      {sellsInLocalMarket}
                    </Text>
                    <Ionicons 
                      name={showMarketDropdown ? 'chevron-up' : 'chevron-down'} 
                      size={20} 
                      style={styles.dropdownIcon} 
                    />
                  </TouchableOpacity>
                  
                  {showMarketDropdown && (
                    <View style={styles.dropdownContainer}>
                      <TouchableOpacity 
                        style={styles.dropdownItem}
                        onPress={() => {
                          setSellsInLocalMarket('Yes');
                          setShowMarketDropdown(false);
                        }}
                      >
                        <Text>Yes</Text>
                      </TouchableOpacity>
                      <TouchableOpacity 
                        style={styles.dropdownItem}
                        onPress={() => {
                          setSellsInLocalMarket('No');
                          setShowMarketDropdown(false);
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
                  style={[styles.button, styles.nextButton]}
                  onPress={handleNext}
                >
                  <Text style={[styles.buttonText, styles.nextButtonText]}>Next</Text>
                </TouchableOpacity>
              </View>
            </View>
          </ScrollView>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>
    </View>
  );
}