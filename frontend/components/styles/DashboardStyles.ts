import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9f9f9',
    padding: 16,
  },
  header: {
    marginTop: 40,
    marginBottom: 16,
  },
  welcome: {
    fontSize: 22,
    color: '#3A3502',
  },
  bold: {
    fontWeight: 'bold',
    color: '#3A3502',
  },
  subHeader: {
    fontSize: 20,
    color: '#3A3502',
    marginBottom: 12,
  },
  weatherCard: {
    backgroundColor: '#E6E6DA',
    borderRadius: 16,
    padding: 16,
    marginBottom: 20,
  },
  weatherRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  weatherImage: {
    width: 60,
    height: 60,
  },
  weatherDegree: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
  },
  feelsLike: {
    fontSize: 14,
    color: '#666',
    marginBottom: 10,
    textAlign: 'center',
  },
  dateText: {
    fontSize: 14,
    color: '#3A3502',
    marginBottom: 6,
    fontWeight: '600',
  },
  profileCard: {
    backgroundColor: '#E8F4D9',
    borderRadius: 16,
    padding: 16,
    marginBottom: 20,
  },
  profileRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  profileIcon: {
    width: 40,
    height: 40,
    marginRight: 12,
  },
  profileText: {
    fontSize: 14,
    color: '#333',
  },
  progressBarBackground: {
    height: 6,
    backgroundColor: '#ccc',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 10,
  },
  progressBarFill: {
    height: 6,
    backgroundColor: '#3A3502',
  },
  profileButton: {
    backgroundColor: '#3A3502',
    borderRadius: 8,
    padding: 8,
    alignItems: 'center',
  },
  profileButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  landsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  landsTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
  },
  noLandText: {
    fontSize: 14,
    color: '#999',
    marginBottom: 20,
  },
  benefitsCard: {
    backgroundColor: '#F5F5E7',
    borderRadius: 16,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 24,
  },
  farmerImage: {
    width: 90, // increased size
    height: 90,
    marginRight: 16,
  },
  benefitsList: {
    flex: 1,
  },
  benefitItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  benefitText: {
    fontSize: 14,
    marginLeft: 8,
    color: '#333',
  },
  storeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
    alignItems: 'center',
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
  },
  viewMore: {
    color: '#3A3502',
    fontSize: 14,
  },
  storeGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  productCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 12,
    width: '48%',
    marginBottom: 16,
    elevation: 2,
  },
  productImage: {
    width: '100%',
    height: 80,
    borderRadius: 8,
    marginBottom: 8,
  },
  productTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  productPrice: {
    fontSize: 12,
    color: '#555',
  },
  productType: {
    fontSize: 12,
    color: '#777',
  },
  productActions: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 8,
    gap: 12,
  },
  quantityText: {
    fontSize: 16,
    color: '#3A3502',
    fontWeight: '600',
  },
  aiCard: {
    backgroundColor: '#E6F0E2',
    padding: 16,
    borderRadius: 16,
    marginBottom: 20,
  },
  aiFeature: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  aiText: {
    fontSize: 14,
    marginLeft: 8,
    color: '#333',
  },
  askNowButton: {
    backgroundColor: '#3A3502',
    borderRadius: 8,
    padding: 10,
    alignItems: 'center',
    marginTop: 12,
  },
  askNowText: {
    color: '#fff',
    fontWeight: '600',
  },
  tipBox: {
    backgroundColor: '#DDF5D6',
    borderRadius: 12,
    padding: 12,
    marginBottom: 80, // space above bottom nav
  },
  tipLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  tipText: {
    fontSize: 14,
    color: '#333',
    marginTop: 4,
  },
  bottomNav: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: '#3A3502',
    paddingVertical: 10,
  },
  navItem: {
    alignItems: 'center',
  },
  navLabel: {
    fontSize: 12,
    marginTop: 4,
    color: '#dcdcdc',
  },
  navLabelActive: {
    color: '#ffffff',
    fontWeight: 'bold',
  },
});
