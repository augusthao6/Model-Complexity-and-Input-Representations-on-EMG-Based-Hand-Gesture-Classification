#Define subject datasets to evaluate
subjects = list(range(1,28))

#Define lists to store data accumulated across multiple subjects
all_acc = []
all_train_time = []
all_test_time = []
all_train_latency = []
all_test_latency = []
all_test_loss = []

#Define learning rate scheduler
def scheduler(epoch):
    if epoch < 50:
        return 0.001
    elif 50 <= epoch < 100:
        return 0.0005
    elif 100 <= epoch < 150:
        return 0.0001
    else:
        return 0.00001


#For plotting average loss and accuracy over all subjects
epoch_metrics = defaultdict(lambda: {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Iterate through subjects
for subject in subjects:

    #Retreive input sEMG feature data and target labels from CSV files
    #CHANGE THIS to get different types of data when needed
    inputs, targets = inputstargets(subject,"train")
    val_inputs, val_targets = inputstargets(subject,"validation")
    test_inputs, test_targets = inputstargets(subject,"test")

    inputs = np.array(inputs)
    val_inputs = np.array(val_inputs)
    test_inputs = np.array(test_inputs)

    if inputs.ndim == 3:
        inputs = inputs.transpose(0, 2, 1)
        val_inputs = val_inputs.transpose(0, 2, 1)
        test_inputs = test_inputs.transpose(0, 2, 1)

    inputs = torch.tensor(inputs, dtype=torch.float32)
    val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
    test_inputs = torch.tensor(test_inputs, dtype=torch.float32)

    targets = torch.tensor(targets, dtype=torch.long)
    val_targets = torch.tensor(val_targets, dtype=torch.long)
    test_targets = torch.tensor(test_targets, dtype=torch.long)

    train_dataset = TensorDataset(inputs, targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    #Initiate type of model - CHANGE THIS to test different models
    model = middleCNN(input_channels=inputs.shape[1], num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_time_start = time.time()

    #Training loop
    for epoch in range(200):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        lr = scheduler(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"\n Epoch {epoch+1}: Learning Rate = {lr}")

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(yb).sum().item()
            total += yb.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)

                val_loss += loss.item() * xb.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(yb).sum().item()
                val_total += yb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Subject {subject} | Epoch {epoch+1}/200 | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


        epoch_metrics[epoch]['loss'].append(train_loss)
        epoch_metrics[epoch]['val_loss'].append(val_loss)
        epoch_metrics[epoch]['accuracy'].append(train_acc)
        epoch_metrics[epoch]['val_accuracy'].append(val_acc)

    train_time = time.time() - train_time_start
    model.eval()
    correct, total = 0, 0
    test_loss = 0
    Actual_Class, Predicted_Class = [], []

    test_start_time = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            test_loss += loss.item() * xb.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(yb).sum().item()
            total += yb.size(0)

            Predicted_Class.extend(predicted.cpu().tolist())
            Actual_Class.extend(yb.cpu().tolist())

    test_time = time.time() - test_start_time
    test_loss /= total

    print(f"Subject {subject} | Test Accuracy: {correct / total:.4f} | Test Loss: {test_loss:.4f}")


    all_acc.append(correct / total)
    all_train_time.append(train_time)
    all_test_time.append(test_time)
    all_test_loss.append(test_loss)

    train_latency = train_time / len(train_dataset)
    test_latency = test_time / len(test_dataset)
    all_train_latency.append(train_latency)
    all_test_latency.append(test_latency)

    print(classification_report(Actual_Class, Predicted_Class))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(Actual_Class, Predicted_Class), display_labels=[str(i) for i in range(10)])
    disp.plot(cmap="Blues")
    plt.show()

# For the following files: CHANGE NAMES for clarity of task, input data, and model
#Save individual test results to a CSV 
with open("individual_test_results_TD_10_CNN1.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Subject', 'Test Accuracy', 'Test Loss', 'Test Time', 'Train Latency', 'Test Latency'])

    for i, subject in enumerate(subjects):
        writer.writerow([subject, all_acc[i], all_test_loss[i], all_test_time[i], all_train_latency[i], all_test_latency[i]])

print("Individual test results have been saved to 'individual_test_results_TD_10_CNN1.csv'.")

#Average metrics per epoch
avg_metrics = {
    'loss': [np.mean(epoch_metrics[e]['loss']) for e in sorted(epoch_metrics)],
    'val_loss': [np.mean(epoch_metrics[e]['val_loss']) for e in sorted(epoch_metrics)],
    'accuracy': [np.mean(epoch_metrics[e]['accuracy']) for e in sorted(epoch_metrics)],
    'val_accuracy': [np.mean(epoch_metrics[e]['val_accuracy']) for e in sorted(epoch_metrics)],
}

avg_metrics_df = pd.DataFrame(avg_metrics)
avg_metrics_df.to_csv("avg_metrics_per_epoch_TD_10_CNN1.csv", index_label="epoch")

!cp /content/avg_metrics_per_epoch_TD_10_CNN1.csv /content/drive/MyDrive/Models
!cp /content/individual_test_results_TD_10_CNN1.csv /content/drive/MyDrive/Models

plt.figure()
plt.plot(avg_metrics_df['loss'], label='Avg Training Loss')
plt.plot(avg_metrics_df['val_loss'], label='Avg Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Average Loss vs Epoch")
plt.show()

plt.figure()
plt.plot(avg_metrics_df['accuracy'], label='Avg Training Acc')
plt.plot(avg_metrics_df['val_accuracy'], label='Avg Validation Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Average Accuracy vs Epoch")
plt.show()