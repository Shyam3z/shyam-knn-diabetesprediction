import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Visualize the data
def data_visualization(data):
    st.subheader("Data Visualization")
    diabetic_patients = data[data.Outcome == 1]
    healthy_individuals = data[data.Outcome == 0]
    
    

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(healthy_individuals.Age, healthy_individuals.Glucose, color="green", label="Healthy", alpha=0.4)
    ax.scatter(diabetic_patients.Age, diabetic_patients.Glucose, color="red", label="Diabetic Patient", alpha=0.4)
    ax.set_xlabel("Age")
    ax.set_ylabel("Glucose")
    ax.legend()
    st.pyplot(fig)

    st.write("### Pair Plot")
    pair_plot = sns.pairplot(data, hue='Outcome', palette='husl')
    st.pyplot(pair_plot)
    
    st.write("### Age Distribution by Outcome")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Outcome', y='Age', data=data, palette='Set2')
    st.pyplot()
    
    st.write("### Glucose Level Distribution by Outcome")
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Outcome', y='Glucose', data=data, palette='Set3')
    st.pyplot()

    st.write("### Age Distribution")
    plt.figure(figsize=(8, 6))
    sns.histplot(data['Age'], bins=20, kde=True, color='skyblue')
    st.pyplot()

    st.write("### Glucose Level Distribution")
    plt.figure(figsize=(8, 6))
    sns.histplot(data['Glucose'], bins=20, kde=True, color='salmon')
    st.pyplot()

    st.write("### BMI Distribution")
    plt.figure(figsize=(8, 6))
    sns.histplot(data['BMI'], bins=20, kde=True, color='green')
    st.pyplot()
# Visualize body to represent diabetes risk factors
def visualize_body(diabetes_risk_factors):
    # Define the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the body outline
    ax.add_patch(patches.Rectangle((0.2, 0.1), 0.6, 0.8, linewidth=2, edgecolor='black', facecolor='none'))

    # Add labels for organs or regions
    ax.text(0.5, 0.95, "Head", ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.8, "Chest", ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.5, "Abdomen", ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.2, "Legs", ha='center', va='center', fontsize=12, fontweight='bold')

    # Define marker sizes
    marker_size = 1000

    # Add markers for diabetes risk factors
    for factor in diabetes_risk_factors:
        if factor == 'Obesity':
            ax.scatter(0.5, 0.5, s=marker_size, color='red', alpha=0.5, marker='o')
        elif factor == 'High Blood Pressure':
            ax.scatter(0.5, 0.45, s=marker_size, color='red', alpha=0.5, marker='s')
        elif factor == 'High Glucose':
            ax.scatter(0.5, 0.4, s=marker_size, color='red', alpha=0.5, marker='^')
        elif factor == 'High Cholesterol':
            ax.scatter(0.5, 0.35, s=marker_size, color='red', alpha=0.5, marker='D')

    # Set axis limits and hide axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    return fig