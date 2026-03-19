BATTERY CELL GROUOINF ENGINNE
This repository contains a high-performance Python engine designed to group battery cells into packs based on their electrochemical profiles. Originally developed for CL05 and CL76 workflows, the tool ensures pack longevity by minimizing variance in capacity and internal resistance (IR).

KEY FEATURES
The engine leverages modern Python patterns to ensure reliability and maintainability:

Object-Oriented Architecture: Logic is encapsulated within a BatteryCellGrouper class for better state management.

Data Validation: Uses Pydantic to enforce strict typing and validation on user inputs and configuration.

Outlier Management: Iteratively identifies and removes (or groups) upper and lower capacity outliers to prevent "weak link" pack configurations.

Advanced Clustering: Employs Gaussian Mixture Models (GMM) from scikit-learn to identify optimal cluster centers for cell distribution.

Safety & Logging: Features custom Decorators for performance tracking and Context Managers for resilient file I/O operations.

CONFIGURATION
Parameter	Description
- Capacity Range	The allowable +/- % variance for cell capacity within a single pack.
- IR Range	The allowable +/- % variance for internal resistance.
- Pack Size	The number of individual cells required per battery pack/string.

USAGE
1. Place your cell data in a .csv file.
2. Ensure your CSV includes the following headers: Cell ID, Latest Cycle N1 Discharge Capacity (Ah), and Latest Cycle N1 DCIR (Ohm-cm2).
3. Run the grouping script: python groupCells.py
4. Follow the interactive prompts to define your tolerance ranges and outlier handling preferences.

ROADMAP

We are actively evolving this engine to handle more complex battery assembly requirements:

1. Multi-Variable Expansion
We are expanding the clustering algorithm to support 3 or 4 variables simultaneously:
- Capacity (Ah)
- Internal Resistance (IR)
- Open Circuit Voltage (OCV)
- Cycle Count History

2. QuickBase Integration
- Direct API integration to update cell group assignments in QuickBase.
- Elimination of manual CSV uploads for production line tracking.

LICENSE

This project is intended for internal battery manufacturing and R&D workflows.
