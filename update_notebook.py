import json

with open('main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Expo Classification Model\n",
        "\n",
        "Here we train a specific model using only features that can be easily asked or measured in a live Expo environment (e.g. age, height, weight, sprint speed, jumping, strength, stamina, shot power).\n",
        "\n",
        "We want to see how accurately we can predict the position with just these physical and basic attributes."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from sklearn.svm import SVC\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import classification_report, confusion_matrix\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "expo_features = ['age', 'height_cm', 'weight_kgs', 'sprint_speed', \n",
        "                 'jumping', 'shot_power', 'stamina', 'strength']\n",
        "\n",
        "# Ensure we drop NA values for these features\n",
        "df_expo = df_clean.dropna(subset=expo_features + ['position_class']).copy()\n",
        "X_expo = df_expo[expo_features].values\n",
        "y_expo = df_expo['position_class'].values\n",
        "\n",
        "# Train/Test split\n",
        "X_train_ex, X_test_ex, y_train_ex, y_test_ex = train_test_split(\n",
        "    X_expo, y_expo, test_size=0.2, random_state=42, stratify=y_expo\n",
        ")\n",
        "\n",
        "# Scale\n",
        "scaler_ex = StandardScaler()\n",
        "X_train_scaled = scaler_ex.fit_transform(X_train_ex)\n",
        "X_test_scaled = scaler_ex.transform(X_test_ex)\n",
        "\n",
        "# Train SVM\n",
        "svm_ex = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', random_state=42)\n",
        "svm_ex.fit(X_train_scaled, y_train_ex)\n",
        "\n",
        "# Predict and Evaluate\n",
        "y_pred_ex = svm_ex.predict(X_test_scaled)\n",
        "print(\"Expo Model Classification Report:\\n\")\n",
        "print(classification_report(y_test_ex, y_pred_ex))\n",
        "\n",
        "# Plot confusion matrix\n",
        "cm = confusion_matrix(y_test_ex, y_pred_ex, labels=svm_ex.classes_)\n",
        "plt.figure(figsize=(6, 5))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
        "            xticklabels=svm_ex.classes_, yticklabels=svm_ex.classes_)\n",
        "plt.title('Confusion Matrix - Expo Features Only')\n",
        "plt.ylabel('True Class')\n",
        "plt.xlabel('Predicted Class')\n",
        "plt.show()\n"
    ]
}

nb['cells'].append(markdown_cell)
nb['cells'].append(code_cell)

with open('main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
