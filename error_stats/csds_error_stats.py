"""
GeolocationStatistics.py is sample code for processing geolocation errors
to produce error statistics...specifically, the performance verification
metric of % of geolocation errors less than 250m.  (System is performing
within spec if >39% of the nadir-equivalent geolocation errors are <250m.)

Geolocation errors are determined through image matching as angular
errors in latitude and longitude, expressed in degrees.  The code below
performs the following steps:
1) Convert the angular errors to N-S distance, and E-W distance [km]
2) Transform the error distance components to view-plane,
   cross-view-plane distances
3) Scale the view-plane and cross-view-plane distance errors to nadir-
   equivalent
4) Analyze sets of these scaled errors statistically
"""

import numpy as np
from typing import Dict


class GeolocationStatistics:
    """Class for processing geolocation errors and computing statistics."""
    
    def __init__(self, num_errors: int = 13):
        """Initialize with specified number of error measurements."""
        self.num_errors = num_errors
        self.RE = 6378140.0  # Earth radius in meters
        self._initialize_test_data()
    
    def _initialize_test_data(self) -> None:
        """Initialize the test data set with 13 error measurements."""
        # Initialize arrays
        self.laterrordeg = np.zeros(self.num_errors)
        self.lonerrordeg = np.zeros(self.num_errors)
        self.RISS_CTRS = np.zeros((self.num_errors, 3))
        self.bhat_HS = np.zeros((self.num_errors, 3))
        self.T_HS2CTRS = np.zeros((3, 3, self.num_errors))
        self.CPlatdeg = np.zeros(self.num_errors)
        self.CPlondeg = np.zeros(self.num_errors)
        self.CPalt = np.zeros(self.num_errors)
        
        # Result arrays
        self.err_xvp = np.zeros(self.num_errors)
        self.err_vp = np.zeros(self.num_errors)
        self.th = np.zeros(self.num_errors)
        self.nequiv_err_xvp = np.zeros(self.num_errors)
        self.nequiv_err_vp = np.zeros(self.num_errors)
        self.nequiv_err_total = np.zeros(self.num_errors)
        self.vp_factor = np.zeros(self.num_errors)
        self.xvp_factor = np.zeros(self.num_errors)
        
        # Load test data
        self._load_test_cases()
    
    def _load_test_cases(self) -> None:
        """Load all 13 test cases with their measurement data."""
        
        # Error Measurement 1
        self.laterrordeg[0] = 0.026980
        self.lonerrordeg[0] = -0.027266
        self.RISS_CTRS[0] = [-3888220.86746399, 5466997.0490439, -1000356.92985575]
        self.bhat_HS[0] = [0, 0.0625969755450201, 0.99803888634292]
        self.T_HS2CTRS[:, :, 0] = [
            [-0.418977524967338, 0.748005379751721, 0.514728846515064],
            [-0.421890284446342, 0.341604851993858, -0.839830169131854],
            [-0.804031356019172, -0.569029065124742, 0.172451447025628]
        ]
        self.CPlatdeg[0] = -8.57802802047
        self.CPlondeg[0] = 125.482222317
        self.CPalt[0] = 44
        
        # Error Measurement 2
        self.laterrordeg[1] = -0.0269188
        self.lonerrordeg[1] = 0.018384
        self.RISS_CTRS[1] = [2138128.91767507, -6313660.02871594, 1241996.71916521]
        self.bhat_HS[1] = [0, 0.0313138440396569, 0.999509601340307]
        self.T_HS2CTRS[:, :, 1] = [
            [0.509557370616697, 0.714990103896663, -0.478686157497828],
            [0.336198439435013, 0.346660121582392, 0.875669669125261],
            [0.792036549265032, -0.607137473258174, -0.0637353370903461]
        ]
        self.CPlatdeg[1] = 10.9913499301
        self.CPlondeg[1] = -71.8457829833
        self.CPalt[1] = 4
        
        # Error Measurement 3
        self.laterrordeg[2] = 0.009040
        self.lonerrordeg[2] = 0.010851
        self.RISS_CTRS[2] = [-2836930.06048711, -4869372.01247407, 3765186.91739563]
        self.bhat_HS[2] = [0, 0.000368458389306164, 0.999999932119205]
        self.T_HS2CTRS[:, :, 2] = [
            [0.436608377090994, -0.795688667243495, 0.419824570355571],
            [-0.682818757213707, 0.0107593091164333, 0.730508577680278],
            [-0.585774418911493, -0.605610255930006, -0.53861354240429]
        ]
        self.CPlatdeg[2] = 33.9986324792
        self.CPlondeg[2] = -120.248435967
        self.CPalt[2] = 0
        
        # Error Measurement 4
        self.laterrordeg[3] = -0.008925
        self.lonerrordeg[3] = -0.026241
        self.RISS_CTRS[3] = [5764626.80186185, -843462.027662883, 3457275.08087601]
        self.bhat_HS[3] = [0, -0.0368375032472, 0.999321268839262]
        self.T_HS2CTRS[:, :, 3] = [
            [-0.275228112982228, 0.368161232084539, -0.888091658002842],
            [0.740939532874243, 0.669849578957866, 0.0480640218257623],
            [0.612583132683508, -0.644793648200697, -0.457146646921637]
        ]
        self.CPlatdeg[3] = 31.1629017783
        self.CPlondeg[3] = -8.7815788192
        self.CPalt[3] = 894
        
        # Error Measurement 5
        self.laterrordeg[4] = 0.022515
        self.lonerrordeg[4] = 0.000992
        self.RISS_CTRS[4] = [2210828.23546441, -6156903.77352567, -1818743.37976767]
        self.bhat_HS[4] = [0, -0.0699499255109834, 0.997550503945042]
        self.T_HS2CTRS[:, :, 4] = [
            [0.497596843733441, -0.8343127195548, -0.237317650198193],
            [0.404893735025568, -0.0185495841473054, 0.914175571903453],
            [-0.767110451267327, -0.550979308973609, 0.328577778675617]
        ]
        self.CPlatdeg[4] = -69.6971234613
        self.CPlondeg[4] = -15.2311156066
        self.CPalt[4] = 3925
        
        # Error Measurement 6
        self.laterrordeg[5] = 0.001
        self.lonerrordeg[5] = 0.0005
        self.RISS_CTRS[5] = [4160733.71254889, 3708441.12891715, 3850046.48797648]
        self.bhat_HS[5] = [0, -0.0368375032472, 0.999321268839262]
        self.T_HS2CTRS[:, :, 5] = [
            [-0.765506977045252, 0.0328563789337692, -0.642588135250651],
            [0.295444324153605, 0.905137001368494, -0.305678298647175],
            [0.571587018239969, -0.423847628778339, -0.702596052277786]
        ]
        self.CPlatdeg[5] = 34.2
        self.CPlondeg[5] = 42.5
        self.CPalt[5] = 0
        
        # Error Measurement 7
        self.laterrordeg[6] = -0.0015
        self.lonerrordeg[6] = 0.0015
        self.RISS_CTRS[6] = [4060487.97522754, -4920200.36807653, -2308736.58835498]
        self.bhat_HS[6] = [0, -0.0257892283106606, 0.999667402541035]
        self.T_HS2CTRS[:, :, 6] = [
            [0.629603159973548, 0.368109063956699, -0.684174861189846],
            [0.215915166022854, 0.763032030046674, 0.609230735287836],
            [0.746311263503805, -0.531296841841519, 0.400927218783918]
        ]
        self.CPlatdeg[6] = -19.7
        self.CPlondeg[6] = -51.1
        self.CPalt[6] = 1000
        
        # Error Measurement 8
        self.laterrordeg[7] = -0.002
        self.lonerrordeg[7] = 0.0
        self.RISS_CTRS[7] = [-4274543.69565126, -116765.394831108, -5276242.10262264]
        self.bhat_HS[7] = [0, 0.0257892283106606, 0.999667402541035]
        self.T_HS2CTRS[:, :, 7] = [
            [0.194530273749036, 0.949748975936332, 0.245223854916003],
            [-0.978512013106359, 0.205316430746179, -0.0189577388931897],
            [-0.0683535866042618, -0.236266544212139, 0.969281089206121]
        ]
        self.CPlatdeg[7] = -49.5
        self.CPlondeg[7] = -178.3
        self.CPalt[7] = 500
        
        # Error Measurement 9
        self.laterrordeg[8] = 0.002
        self.lonerrordeg[8] = -0.0005
        self.RISS_CTRS[8] = [2520101.83352962, 6230331.37805726, -961492.530214298]
        self.bhat_HS[8] = [0, 0.0625969755450201, 0.99803888634292]
        self.T_HS2CTRS[:, :, 8] = [
            [-0.446421529583839, 0.413968410219497, -0.793307812225063],
            [0.384674732015686, -0.711668847399292, -0.587837230750712],
            [-0.807919035840032, -0.56758835193185, 0.158460875505101]
        ]
        self.CPlatdeg[8] = -8.2
        self.CPlondeg[8] = 70.0
        self.CPalt[8] = 500
        
        # Error Measurement 10
        self.laterrordeg[9] = 0.001
        self.lonerrordeg[9] = 0.001
        self.RISS_CTRS[9] = [4248835.7920035, -2447631.1800248, 4676942.35070364]
        self.bhat_HS[9] = [0, 0.0552406262884485, 0.998473070847311]
        self.T_HS2CTRS[:, :, 9] = [
            [0.632159685228781, -0.204512480192889, -0.747361135669863],
            [0.598189792213041, -0.48424547358001, 0.638493680968301],
            [-0.492486654503011, -0.850693598485042, -0.183784021560657]
        ]
        self.CPlatdeg[9] = 46.0
        self.CPlondeg[9] = -29.0
        self.CPalt[9] = 50
        
        # Error Measurement 11
        self.laterrordeg[10] = -0.001
        self.lonerrordeg[10] = -0.0015
        self.RISS_CTRS[10] = [-5515282.275281, -925908.369707886, 3822512.18293707]
        self.bhat_HS[10] = [0, -0.0147378023382108, 0.999891392693346]
        self.T_HS2CTRS[:, :, 10] = [
            [0.753428906287479, -0.49153961754033, 0.436730605865421],
            [-0.565149851875981, -0.823589060852856, 0.0480236900131712],
            [0.336081133202996, -0.283000352923251, -0.898309519920648]
        ]
        self.CPlatdeg[10] = 32.5
        self.CPlondeg[10] = -170.4
        self.CPalt[10] = 100
        
        # Error Measurement 12
        self.laterrordeg[11] = 0.0011
        self.lonerrordeg[11] = 0.0011
        self.RISS_CTRS[11] = [-824875.850002718, 6312906.79811629, -2344264.22196647]
        self.bhat_HS[11] = [0, -0.0221057030926655, 0.999755639089262]
        self.T_HS2CTRS[:, :, 11] = [
            [-0.585265557251293, -0.595045400433036, 0.550803349451662],
            [-0.109341614649782, -0.615175192945938, -0.780771245706364],
            [0.803435522491452, -0.517183787785386, 0.294977548217097]
        ]
        self.CPlatdeg[11] = -20.33
        self.CPlondeg[11] = 95.4
        self.CPalt[11] = 100
        
        # Error Measurement 13
        self.laterrordeg[12] = 0.0025
        self.lonerrordeg[12] = 0.0
        self.RISS_CTRS[12] = [3675746.11507236, -2198122.65541618, -5270960.3157354]
        self.bhat_HS[12] = [0, -0.0221057030926655, 0.999755639089262]
        self.T_HS2CTRS[:, :, 12] = [
            [0.292122841971449, -0.95622050459562, 0.017506859615382],
            [0.95633436246494, 0.291879296125504, -0.0152004494429911],
            [0.00942509242927969, 0.0211828985704245, 0.999731155222533]
        ]
        self.CPlatdeg[12] = -47.6
        self.CPlondeg[12] = -30.87
        self.CPalt[12] = 100
    
    def _convert_inputs(self) -> None:
        """Convert angular errors to distances and transform coordinates."""
        self.laterrorrad = self.laterrordeg * np.pi / 180
        self.lonerrorrad = self.lonerrordeg * np.pi / 180
        self.CPlatrad = self.CPlatdeg * np.pi / 180
        self.CPlonrad = self.CPlondeg * np.pi / 180
        
        # Transform bhat from HS to CTRS coordinate system
        self.bhat_CTRS = np.zeros((self.num_errors, 3))
        for i in range(self.num_errors):
            T = self.T_HS2CTRS[:, :, i]
            self.bhat_CTRS[i] = self.bhat_HS[i] @ T.T
        
        # Calculate N-S and E-W error distances in meters (to match MATLAB)
        self.NSerrdist = self.RE * self.laterrorrad  # Keep in meters
        self.EWerrdist = self.RE * np.cos(self.CPlatrad) * self.lonerrorrad  # Keep in meters

    def _process_errors_to_nadir_equivalent(self) -> None:
        """Process all measured errors to nadir-equivalent."""
        for i in range(self.num_errors):
            # Calculate transformation from CTRS to up-east-north (UEN)
            T_CTRS2UEN = np.zeros((3, 3))
            T_CTRS2UEN[0] = [
                np.cos(self.CPlonrad[i]) * np.cos(self.CPlatrad[i]),
                np.sin(self.CPlonrad[i]) * np.cos(self.CPlatrad[i]),
                np.sin(self.CPlatrad[i])
            ]
            T_CTRS2UEN[1] = [
                -np.sin(self.CPlonrad[i]),
                np.cos(self.CPlonrad[i]),
                0
            ]
            T_CTRS2UEN[2] = [
                -np.cos(self.CPlonrad[i]) * np.sin(self.CPlatrad[i]),
                -np.sin(self.CPlonrad[i]) * np.sin(self.CPlatrad[i]),
                np.cos(self.CPlatrad[i])
            ]
            
            # Transform bhat from CTRS to UEN
            bhat_UEN = self.bhat_CTRS[i] @ T_CTRS2UEN.T
            
            # Calculate V_UEN and X_UEN unit vectors
            norm_factor = np.sqrt(bhat_UEN[1]**2 + bhat_UEN[2]**2)
            V_UEN = np.array([0, bhat_UEN[1], bhat_UEN[2]]) / norm_factor
            X_UEN = np.array([0, bhat_UEN[2], -bhat_UEN[1]]) / norm_factor
            
            # Create transformation matrix from UEN to UXV
            T_UEN2UXV = np.eye(3)
            T_UEN2UXV[1] = X_UEN
            T_UEN2UXV[2] = V_UEN
            
            # Transform error distances from East/North to X-view-plane/view-plane
            temp = np.array([0, self.EWerrdist[i], self.NSerrdist[i]]) @ T_UEN2UXV.T
            self.err_xvp[i] = temp[1]
            self.err_vp[i] = temp[2]
            
            # Scale these error distances to nadir-equivalent
            Rhat = self.RISS_CTRS[i] / np.linalg.norm(self.RISS_CTRS[i])
            bhat_norm = np.linalg.norm(self.bhat_CTRS[i])
            Rhat_norm = np.linalg.norm(Rhat)
            self.th[i] = np.arccos(np.dot(self.bhat_CTRS[i], -Rhat) / (bhat_norm * Rhat_norm))

            f = np.linalg.norm(self.RISS_CTRS[i]) / self.RE
            h = np.linalg.norm(self.RISS_CTRS[i]) - self.RE
            temp1 = np.sqrt(1 - f**2 * np.sin(self.th[i])**2)
            
            self.xvp_factor[i] = h / self.RE / np.cos(self.th[i]) / (f * np.cos(self.th[i]) - temp1)
            self.vp_factor[i] = h / self.RE / (-1 + f * np.cos(self.th[i]) / temp1)
            
            self.nequiv_err_xvp[i] = self.err_xvp[i] * self.xvp_factor[i]
            self.nequiv_err_vp[i] = self.err_vp[i] * self.vp_factor[i]
            self.nequiv_err_total[i] = np.sqrt(self.nequiv_err_xvp[i]**2 + self.nequiv_err_vp[i]**2)
    
    def _display_results(self) -> None:
        """Display the processing results in formatted tables."""
        print("        VP error             Off-nadir ang          VP factor         NadEquiv VP error")
        for i in range(self.num_errors):
            print(f"{self.err_vp[i]:15.6f} {self.th[i]*180/np.pi:20.6f} {self.vp_factor[i]:15.6f} {self.nequiv_err_vp[i]:20.6f}")
        
        print("\n        XVP error            Off-nadir ang         XVP factor          NadEquiv XVP error")
        for i in range(self.num_errors):
            print(f"{self.err_xvp[i]:15.6f} {self.th[i]*180/np.pi:20.6f} {self.xvp_factor[i]:15.6f} {self.nequiv_err_xvp[i]:20.6f}")
        
        print("\n     NadEquiv VP error         NadEquiv XVP error      NadEquive total error")
        for i in range(self.num_errors):
            print(f"{self.nequiv_err_vp[i]:20.6f} {self.nequiv_err_xvp[i]:20.6f} {self.nequiv_err_total[i]:20.6f}")
    
    def calculate_statistics(self) -> Dict[str, float]:
        """Calculate statistics on nadir-equivalent errors."""
        # nequiv_err_total is already in meters (to match MATLAB)
        nequiv_err_total_m = self.nequiv_err_total  # Already in meters, no conversion needed

        # Count errors less than 250m
        num_less250 = np.sum(nequiv_err_total_m < 250)
        
        # Calculate mean error distance
        mean_err_dist = np.mean(nequiv_err_total_m)
        
        # Calculate percentage less than 250m
        pct_err_less250 = (num_less250 / self.num_errors) * 100
        
        return {
            'mean_error_distance_m': mean_err_dist,
            'percent_less_than_250m': pct_err_less250,
            'num_less_than_250m': num_less250,
            'total_measurements': self.num_errors
        }
    
    def run_analysis(self, display_results: bool = True) -> Dict[str, float]:
        """Run the complete geolocation error analysis."""
        print(f"Processing {self.num_errors} geolocation error measurements...")
        
        # Convert inputs
        self._convert_inputs()
        
        # Process errors to nadir-equivalent
        self._process_errors_to_nadir_equivalent()
        
        # Display detailed results if requested
        if display_results:
            self._display_results()
        
        # Calculate and display statistics
        stats = self.calculate_statistics()
        
        print(f"\n--- STATISTICS ---")
        print(f"Mean error distance: {stats['mean_error_distance_m']:.2f} m")
        print(f"Percentage of errors < 250m: {stats['percent_less_than_250m']:.1f}%")
        print(f"Number of errors < 250m: {stats['num_less_than_250m']} out of {stats['total_measurements']}")
        
        # Performance check
        if stats['percent_less_than_250m'] > 39:
            print("✓ System is performing within spec (>39% of errors < 250m)")
        else:
            print("✗ System is NOT performing within spec (≤39% of errors < 250m)")
        
        return stats


def main():
    """Main function to run the geolocation statistics analysis."""
    # Create analyzer with all 13 error measurements
    analyzer = GeolocationStatistics(num_errors=13)
    
    # Run the analysis
    results = analyzer.run_analysis()
    
    return results


if __name__ == "__main__":
    main()
