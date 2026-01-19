import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import random
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import os

#path for saved model
MODEL_PATH = "brain_classifier.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transforms & dataloaders
def get_dataloaders():
    train_tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dl = DataLoader(
        datasets.ImageFolder('dataset/Training',train_tf),
        batch_size=32, shuffle=True, num_workers=4, pin_memory=False
    )

    test_dl = DataLoader(
        datasets.ImageFolder('dataset/Testing',test_tf),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=False
    )

    return train_dl, test_dl, test_tf

#model
def build_model():
    #define model
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128),  nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 4) #4 classes
    )

    return model.to(device)

#load previously trained model
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    return

#training
def train(model, train_dl, optimizer, loss_fn):
    model.train()
    
    for epoch in range(25):
        running_loss = 0.0 

        for x, y in train_dl:
            optimizer.zero_grad()
            
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x),y)
            loss.backward()

            running_loss += loss.item()

            optimizer.step()
        
        #average loss per batch
        print(f"Epoch {epoch+1}: Loss was {running_loss/ len(train_dl):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return

#evaluation
def evaluate(model, test_dl, loss_fn):
    model.eval()
    test_loss, correct = 0.0, 0

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            test_loss += loss_fn(logits, y).item() * y.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds==y).sum().item()

    test_loss /= len(test_dl.dataset)
    accuracy = 100.0 * correct / len(test_dl.dataset)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.2f}%')

    return accuracy

#visual test sample
def show_test_sample(model, test_dl):
    idx = random.randrange(len(test_dl.dataset))
    img, label = test_dl.dataset[idx]

    unnorm = img * 0.5 + 0.5
    plt.imshow(to_pil_image(unnorm))
    plt.axis('off')
    plt.title("Sample from test set")
    plt.show()

    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        pred = logits.argmax(1).item()

    class_names = test_dl.dataset.classes
    print(f"Predicted class: {class_names[pred]}")
    print(f"Ground-truth: {class_names[label]}")

#single image prediction
def predict_image(img_path, model, transform, class_names):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img)
        pred = logits.argmax(1).item()

    return class_names[pred]

def main():
    train_dl, test_dl, tf = get_dataloaders()
    model = build_model()

    opt = optim.AdamW(model.parameters(), 1e-4)
    loss_fn = nn.CrossEntropyLoss()

    if os.path.exists(MODEL_PATH):
        choice = input("Saved model found. Load it? (y/n): ").lower()
        if choice == "y":
            load_model(model, MODEL_PATH, device)
        else:
            train(model, train_dl, opt, loss_fn)
    else:
        train(model, train_dl, opt, loss_fn)

    run_eval = input("Run evaluation on test set? (y/n): ").lower()
    if run_eval == "y":
        evaluate(model, test_dl, loss_fn)
        show_test_sample(model, test_dl)

    class_names = train_dl.dataset.classes

    #allow user to predict their own image files
    while True:
        img_path = input("\nEnter path to an image (or enter to quit): ")

        if img_path == "":
            print("Exiting")
            break
            
        if not os.path.isfile(img_path):
            print("File not found")
            continue

        pred = predict_image(img_path, model, tf, class_names)
        print(f"Prediction: {pred}")
        

if __name__ == '__main__':
    main()
