# Spec Delta: Traditional AOI Practice

## ADDED Requirements

### Requirement: Traditional AOI baseline candidate generation

系统 SHALL provide a command-line OpenCV baseline that reads one or more images, applies ROI cropping, illumination normalization, threshold or edge processing, connected component analysis, and writes defect candidate regions.

#### Scenario: Generate candidates for one image

- Given a readable input image and a valid ROI configuration
- When the learner runs the traditional AOI baseline command
- Then the system writes a binary mask image
- And the system writes an overlay image with candidate boxes
- And the system writes `candidates.json` with bbox, area, centroid, and source image path

#### Scenario: Reject invalid ROI

- Given an input image with known width and height
- And an ROI configuration outside the image bounds
- When the learner runs the traditional AOI baseline command
- Then the system exits with a clear error message containing the image size and invalid ROI

### Requirement: Camera calibration and undistortion practice

系统 SHALL provide a command-line OpenCV calibration script that estimates camera intrinsics and distortion coefficients from chessboard images and writes undistorted outputs.

#### Scenario: Calibrate from valid chessboard images

- Given a directory containing enough chessboard calibration images
- And a configuration with board inner-corner size and square size in millimeters
- When the learner runs the camera calibration command
- Then the system writes `calibration_result.yaml`
- And the result includes camera matrix, distortion coefficients, reprojection error, valid image count, skipped image count, and approximate pixel/mm information
- And the system writes undistorted images for valid inputs

#### Scenario: Fail clearly with too few valid calibration images

- Given a directory where fewer than the minimum required images contain detectable chessboard corners
- When the learner runs the camera calibration command
- Then the system exits with a clear message explaining that more valid chessboard images are required

### Requirement: Normal sample statistics practice

系统 SHALL provide a command-line analysis script that computes mean, variance, standard deviation, and diff visualizations for a set of normal samples.

#### Scenario: Generate normal sample statistics

- Given a directory containing readable normal sample images of compatible size
- When the learner runs the normal sample statistics command
- Then the system writes `mean.png`, `variance.png`, and `std.png`
- And the system writes per-image diff overlays
- And the system writes `summary.json` with sample count, average std, P95 diff, and P99 diff

#### Scenario: Detect incompatible sample sizes

- Given normal sample images with inconsistent dimensions
- And no resize option configured
- When the learner runs the normal sample statistics command
- Then the system exits with a clear error explaining the dimension mismatch and how to configure resize
