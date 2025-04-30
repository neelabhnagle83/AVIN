import type { NativeStackNavigationProp } from '@react-navigation/native-stack';

export type PhoneLoginNavigationProp = NativeStackNavigationProp<any>;

export interface CountryPickerProps {
  selectedCode: string;
  onSelectCode: (code: string) => void;
}
