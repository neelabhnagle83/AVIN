import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import type { CountryPickerProps } from '../styles/PhoneLogin.types';

const CountryPicker: React.FC<CountryPickerProps> = ({ selectedCode, onSelectCode }) => {
  return (
    <View style={styles.pickerWrapper}>
      <Picker
        selectedValue={selectedCode}
        onValueChange={onSelectCode}
        style={styles.picker}
        dropdownIconColor="#444702"
      >
        <Picker.Item label="+91" value="+91" />
        <Picker.Item label="+1" value="+1" />
        <Picker.Item label="+44" value="+44" />
        <Picker.Item label="+61" value="+61" />
        <Picker.Item label="+81" value="+81" />
        <Picker.Item label="+971" value="+971" />
      </Picker>
    </View>
  );
};

const styles = StyleSheet.create({
    pickerWrapper: {
        width: 110,
        borderWidth: 1,
        borderColor: '#ccc',
        borderRadius: 8,
        marginRight: 5,
        overflow: 'hidden',
        justifyContent: 'center',
      },
  picker: {
    height: 50,
    width: 120,
    color: '#444702',
    marginLeft: -10,
  },
});

export default CountryPicker;
