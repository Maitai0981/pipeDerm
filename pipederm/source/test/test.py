import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import os
import json
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importações específicas do projeto
from transformers import AutoImageProcessor, AutoModelForImageClassification, Blip2Processor, Blip2ForConditionalGeneration
from app.utils import aplicar_clahe
from config import CLASS_MAPPING, MODEL_CONFIG

# =============================================================================
# 1. DATASET ADAPTADO PARA O PROJETO
# =============================================================================

class SkinLesionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augment_quality=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment_quality = augment_quality
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        # Aplicar CLAHE como no projeto real
        image = aplicar_clahe(image)
        
        # Aplicar degradação de qualidade se especificado
        if self.augment_quality:
            image = self.apply_quality_degradation(image, self.augment_quality)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def apply_quality_degradation(self, image, degradation_type):
        """Aplica diferentes tipos de degradação na qualidade da imagem"""
        if degradation_type == 'low_resolution':
            # Simula imagens de celulares antigos
            w, h = image.size
            image = image.resize((w//3, h//3))
            image = image.resize((w, h))
        
        elif degradation_type == 'blur':
            # Simula tremor de mão ou movimento
            image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
        
        elif degradation_type == 'low_lighting':
            # Simula condições amazônicas com pouca luz
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.4)
        
        elif degradation_type == 'high_lighting':
            # Simula flash muito forte ou sol direto
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(2.0)
        
        elif degradation_type == 'humidity_effect':
            # Simula efeito de umidade alta da Amazônia
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(0.7)
            # Adiciona leve névoa
            np_image = np.array(image)
            fog = np.ones_like(np_image) * 20
            np_image = np.clip(np_image + fog, 0, 255).astype(np.uint8)
            image = Image.fromarray(np_image)
        
        elif degradation_type == 'skin_glare':
            # Simula brilho da pele oleosa em clima tropical
            np_image = np.array(image)
            # Cria pontos de brilho aleatórios
            mask = np.random.random(np_image.shape[:2]) > 0.95
            for c in range(3):
                np_image[:, :, c] = np.where(mask, 255, np_image[:, :, c])
            image = Image.fromarray(np_image)
        
        return image

# Transformações otimizadas para o projeto
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =============================================================================
# 2. MODELOS ADAPTADOS DO PROJETO
# =============================================================================

class ProjectSkinClassifier(nn.Module):
    """Wrapper para o modelo NeuronZero/SkinCancerClassifier usado no projeto"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(MODEL_CONFIG["skin_classifier"])
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_CONFIG["skin_classifier"])
        self.model.to(device)
        self.class_mapping = CLASS_MAPPING
        
    def forward(self, x):
        # Se x já é tensor processado, usa direto
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
            with torch.no_grad():
                return self.model(pixel_values=x).logits
        else:
            # Se é imagem PIL, processa primeiro
            inputs = self.processor(images=x, return_tensors="pt").to(self.device)
            with torch.no_grad():
                return self.model(**inputs).logits

class MultimodalDermAI(nn.Module):
    """Sistema multimodal completo baseado no projeto DermAI"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        
        # Componente de classificação (modelo principal)
        self.skin_classifier = ProjectSkinClassifier(device)
        
        # Componente de descrição visual (BLIP-2)
        self.blip2_processor = Blip2Processor.from_pretrained(MODEL_CONFIG["blip2"])
        self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_CONFIG["blip2"], 
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map="auto"
        )
        
        # Camada de fusão multimodal
        self.fusion_layer = nn.Sequential(
            nn.Linear(len(CLASS_MAPPING) + 768, 256),  # Classificação + features textuais
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(CLASS_MAPPING))
        ).to(device)
        
    def get_text_features(self, image):
        """Extrai features textuais usando BLIP-2"""
        prompt = "Descreva as características desta lesão de pele:"
        inputs = self.blip2_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Obtém features do encoder (não gera texto)
            encoder_outputs = self.blip2_model.vision_model(**inputs)
            # Usa pooled output como features (simplificado)
            text_features = encoder_outputs.pooler_output if hasattr(encoder_outputs, 'pooler_output') else encoder_outputs.last_hidden_state.mean(dim=1)
        
        return text_features
    
    def forward(self, x, use_multimodal=True):
        # Classificação visual
        visual_logits = self.skin_classifier(x)
        
        if use_multimodal and not isinstance(x, torch.Tensor):
            # Extrair features textuais (apenas se x for imagem PIL)
            text_features = self.get_text_features(x)
            
            # Fusão multimodal
            combined_features = torch.cat([visual_logits, text_features], dim=1)
            final_logits = self.fusion_layer(combined_features)
            return final_logits
        else:
            return visual_logits

# CNNs clássicas para comparação
class ClassicalCNNs:
    def __init__(self, num_classes=len(CLASS_MAPPING)):
        self.num_classes = num_classes
        
    def get_resnet50(self):
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model
    
    def get_efficientnet_b0(self):
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        return model
    
    def get_densenet121(self):
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
        return model
    
    def get_all_models(self):
        return {
            'ResNet50': self.get_resnet50(),
            'EfficientNet-B0': self.get_efficientnet_b0(),
            'DenseNet121': self.get_densenet121()
        }

# =============================================================================
# 3. SISTEMA DE TREINAMENTO E AVALIAÇÃO
# =============================================================================

class DermAIModelTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        self.class_names = list(CLASS_MAPPING.values())
        
    def train_model(self, model, train_loader, val_loader, model_name, epochs=10):
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_acc = 0.0
        train_losses = []
        val_accuracies = []
        
        print(f"\n🔬 Treinando {model_name}...")
        
        for epoch in range(epochs):
            # Fase de treinamento
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass adaptado para diferentes tipos de modelo
                if isinstance(model, MultimodalDermAI):
                    outputs = model(inputs, use_multimodal=False)  # Use apenas visual durante treinamento
                elif isinstance(model, ProjectSkinClassifier):
                    outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                # Log de progresso
                if batch_idx % 20 == 0:
                    print(f'  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # Fase de validação
            model.eval()
            correct_val = 0
            total_val = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    if isinstance(model, MultimodalDermAI):
                        outputs = model(inputs, use_multimodal=False)
                    elif isinstance(model, ProjectSkinClassifier):
                        outputs = model(inputs)
                    else:
                        outputs = model(inputs)
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            train_acc = 100 * correct_train / total_train
            val_acc = 100 * correct_val / total_val
            epoch_loss = running_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)
            
            train_losses.append(epoch_loss)
            val_accuracies.append(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                # Salvar melhor modelo
                torch.save(model.state_dict(), f'best_{model_name.lower()}.pth')
            
            scheduler.step()
            
            print(f'  Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss_avg:.4f}')
        
        return {
            'best_accuracy': best_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }
    
    def evaluate_model(self, model, test_loader, model_name):
        model.eval()
        model = model.to(self.device)
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if isinstance(model, MultimodalDermAI):
                    outputs = model(inputs, use_multimodal=False)
                elif isinstance(model, ProjectSkinClassifier):
                    outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Contabilizar acertos por classe
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1
        
        # Calcular métricas gerais
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # Calcular métricas por classe
        class_metrics = {}
        for class_idx in range(len(self.class_names)):
            if class_total[class_idx] > 0:
                class_acc = class_correct[class_idx] / class_total[class_idx]
                class_metrics[self.class_names[class_idx]] = {
                    'accuracy': class_acc,
                    'support': class_total[class_idx]
                }
        
        # AUC-ROC (tratamento robusto para classes ausentes)
        try:
            # Verificar se todas as classes estão presentes
            unique_labels = set(all_labels)
            if len(unique_labels) == len(self.class_names):
                auc_score = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='weighted')
            else:
                auc_score = 0.0
                print(f"⚠️ AUC não calculado para {model_name} - classes ausentes no conjunto de teste")
        except Exception as e:
            auc_score = 0.0
            print(f"⚠️ Erro no cálculo de AUC para {model_name}: {e}")
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'class_metrics': class_metrics,
            'predictions': all_predictions,
            'true_labels': all_labels,
            'probabilities': all_probabilities,
            'confusion_matrix': confusion_matrix(all_labels, all_predictions)
        }
        
        return results

# =============================================================================
# 4. TESTES DE ROBUSTEZ ESPECÍFICOS PARA CONTEXTO AMAZÔNICO
# =============================================================================

class AmazonianRobustnessTests:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def test_amazonian_conditions(self, test_images, test_labels):
        """Testa robustez específica para condições amazônicas"""
        amazonian_conditions = [
            'low_resolution',      # Celulares básicos
            'blur',               # Tremor/movimento
            'low_lighting',       # Sombra da floresta
            'high_lighting',      # Sol tropical direto
            'humidity_effect',    # Alta umidade
            'skin_glare'         # Pele oleosa pelo calor
        ]
        
        results = {}
        baseline_acc = self._get_baseline_accuracy(test_images, test_labels)
        
        print("\n🌳 Testando robustez para condições amazônicas...")
        print(f"📊 Acurácia baseline (condições normais): {baseline_acc:.2f}%")
        
        for condition in amazonian_conditions:
            print(f"🔍 Testando condição: {condition}")
            
            # Criar dataset com degradação específica
            dataset = SkinLesionDataset(test_images, test_labels, 
                                      transform=transform_test, 
                                      augment_quality=condition)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            # Avaliar modelo
            accuracy = self._evaluate_degraded_images(loader)
            degradation = baseline_acc - accuracy
            
            results[condition] = {
                'accuracy': accuracy,
                'degradation': degradation,
                'retention_rate': (accuracy / baseline_acc) * 100 if baseline_acc > 0 else 0
            }
            
            print(f"  ✅ Acurácia: {accuracy:.2f}% (↓{degradation:.2f}%, retenção: {results[condition]['retention_rate']:.1f}%)")
        
        return results
    
    def _get_baseline_accuracy(self, test_images, test_labels):
        """Calcula acurácia baseline sem degradação"""
        dataset = SkinLesionDataset(test_images, test_labels, transform=transform_test)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        return self._evaluate_degraded_images(loader)
    
    def _evaluate_degraded_images(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if isinstance(self.model, MultimodalDermAI):
                    outputs = self.model(inputs, use_multimodal=False)
                elif isinstance(self.model, ProjectSkinClassifier):
                    outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total

# =============================================================================
# 5. ANÁLISE DE VIÉS ADAPTADA PARA O CONTEXTO BRASILEIRO
# =============================================================================

class BrazilianBiasAnalysis:
    def __init__(self):
        # Fototipos de Fitzpatrick adaptados para população brasileira
        self.skin_types = {
            'tipo_I_II': 'Pele muito clara/clara',      # Raro no contexto amazônico
            'tipo_III_IV': 'Pele morena/morena escura', # Mais comum
            'tipo_V_VI': 'Pele negra/muito negra'       # Significativa na Amazônia
        }
        
        # Faixas etárias relevantes para câncer de pele no Brasil
        self.age_groups = {
            'jovem': '<35 anos',
            'adulto': '35-60 anos', 
            'idoso': '>60 anos'
        }
    
    def analyze_skin_type_bias(self, model, test_data_by_skin_type):
        """Analisa viés por tipo de pele brasileiro"""
        results = {}
        
        print("\n🔍 Análise de viés por tipo de pele...")
        
        for skin_type, description in self.skin_types.items():
            if skin_type in test_data_by_skin_type:
                images, labels = test_data_by_skin_type[skin_type]
                dataset = SkinLesionDataset(images, labels, transform_test)
                loader = DataLoader(dataset, batch_size=32, shuffle=False)
                
                trainer = DermAIModelTrainer()
                result = trainer.evaluate_model(model, loader, f"SkinType_{skin_type}")
                
                results[skin_type] = {
                    'description': description,
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score'],
                    'sample_count': len(labels)
                }
                
                print(f"  📊 {description}: {result['accuracy']:.3f} acurácia, {len(labels)} amostras")
        
        # Calcular disparidade
        accuracies = [r['accuracy'] for r in results.values()]
        if len(accuracies) > 1:
            max_acc = max(accuracies)
            min_acc = min(accuracies)
            disparity = max_acc - min_acc
            print(f"  ⚠️ Disparidade máxima: {disparity:.3f} ({disparity*100:.1f}%)")
        
        return results
    
    def analyze_regional_bias(self, model, test_data_by_region):
        """Analisa viés por região do Brasil"""
        regions = {
            'norte': 'Norte (Amazônia)',
            'nordeste': 'Nordeste', 
            'centro_oeste': 'Centro-Oeste',
            'sudeste': 'Sudeste',
            'sul': 'Sul'
        }
        
        results = {}
        print("\n🗺️ Análise de viés regional...")
        
        for region, description in regions.items():
            if region in test_data_by_region:
                images, labels = test_data_by_region[region]
                dataset = SkinLesionDataset(images, labels, transform_test)
                loader = DataLoader(dataset, batch_size=32, shuffle=False)
                
                trainer = DermAIModelTrainer()
                result = trainer.evaluate_model(model, loader, f"Region_{region}")
                
                results[region] = {
                    'description': description,
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score'],
                    'sample_count': len(labels)
                }
                
                print(f"  📊 {description}: {result['accuracy']:.3f} acurácia")
        
        return results

# =============================================================================
# 6. ABLATION STUDIES PARA O SISTEMA DERMIA
# =============================================================================

class DermAIAblationStudy:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def test_visual_only(self, train_loader, val_loader, test_loader):
        """Testa apenas o componente de classificação visual"""
        print("\n🔬 Ablation Study: Apenas Classificação Visual")
        
        model = ProjectSkinClassifier(self.device)
        trainer = DermAIModelTrainer(self.device)
        
        # Treinar
        train_result = trainer.train_model(model, train_loader, val_loader, "Visual_Only", epochs=5)
        
        # Avaliar
        test_result = trainer.evaluate_model(model, test_loader, "Visual_Only")
        
        return {
            'component': 'Classificação Visual (NeuronZero/SkinCancerClassifier)',
            'training': train_result,
            'testing': test_result
        }
    
    def test_with_clahe_preprocessing(self, train_loader, val_loader, test_loader):
        """Testa impacto do pré-processamento CLAHE"""
        print("\n🔬 Ablation Study: Com/Sem CLAHE")
        
        # Teste sem CLAHE (modificar dataset temporariamente)
        # Isso requer uma versão modificada do dataset
        model = ProjectSkinClassifier(self.device)
        trainer = DermAIModelTrainer(self.device)
        
        # Avaliar com CLAHE (padrão)
        test_result_with_clahe = trainer.evaluate_model(model, test_loader, "With_CLAHE")
        
        return {
            'component': 'Efeito do pré-processamento CLAHE',
            'with_clahe': test_result_with_clahe,
            # 'without_clahe': test_result_without_clahe  # Implementar se necessário
        }
    
    def test_multimodal_vs_visual(self, train_loader, val_loader, test_loader):
        """Compara sistema multimodal vs apenas visual"""
        print("\n🔬 Ablation Study: Multimodal vs Visual")
        
        # Sistema visual apenas
        visual_model = ProjectSkinClassifier(self.device)
        trainer = DermAIModelTrainer(self.device)
        
        visual_result = trainer.evaluate_model(visual_model, test_loader, "Visual_Component")
        
        # Sistema multimodal completo
        multimodal_model = MultimodalDermAI(self.device)
        multimodal_result = trainer.evaluate_model(multimodal_model, test_loader, "Multimodal_System")
        
        # Calcular ganho multimodal
        improvement = multimodal_result['accuracy'] - visual_result['accuracy']
        
        return {
            'visual_only': visual_result,
            'multimodal': multimodal_result,
            'multimodal_improvement': improvement,
            'improvement_percentage': (improvement / visual_result['accuracy']) * 100 if visual_result['accuracy'] > 0 else 0
        }

# =============================================================================
# 7. VISUALIZAÇÃO ADAPTADA PARA O PROJETO
# =============================================================================

class DermAIResultsVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.class_names = list(CLASS_MAPPING.values())
    
    def plot_model_comparison(self, results_dict):
        """Plota comparação entre modelos com foco em métricas dermatológicas"""
        models = list(results_dict.keys())
        accuracies = [results_dict[model]['accuracy'] for model in models]
        f1_scores = [results_dict[model]['f1_score'] for model in models]
        recalls = [results_dict[model]['recall'] for model in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bars1 = ax.bar(x - width, accuracies, width, label='Acurácia', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x, f1_scores, width, label='F1-Score', alpha=0.8, color='lightcoral')
        bars3 = ax.bar(x + width, recalls, width, label='Recall (Sensibilidade)', alpha=0.8, color='lightgreen')
        
        # Adicionar valores nas barras
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Modelos', fontsize=12)
        ax.set_ylabel('Performance', fontsize=12)
        ax.set_title('Comparação de Performance - Classificação de Lesões de Pele', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_amazonian_robustness(self, robustness_results):
        """Plota resultados de robustez para condições amazônicas"""
        conditions = list(robustness_results.keys())
        accuracies = [robustness_results[cond]['accuracy'] for cond in conditions]
        retention_rates = [robustness_results[cond]['retention_rate'] for cond in conditions]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico 1: Acurácia por condição
        bars1 = ax1.bar(conditions, accuracies, alpha=0.7, color='coral')
        ax1.set_xlabel('Condições Ambientais')
        ax1.set_ylabel('Acurácia (%)')
        ax1.set_title('Robustez em Condições Amazônicas')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Gráfico 2: Taxa de retenção
        bars2 = ax2.bar(conditions, retention_rates, alpha=0.7, color='lightblue')
        ax2.set_xlabel('Condições Ambientais')
        ax2.set_ylabel('Taxa de Retenção (%)')
        ax2.set_title('Taxa de Retenção de Performance')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Limite Aceitável (80%)')
        ax2.legend()
        
        for bar, rate in zip(bars2, retention_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix_detailed(self, y_true, y_pred, model_name):
        """Plota matriz de confusão detalhada com métricas por classe"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalizar matriz de confusão
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Matriz absoluta
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_xlabel('Predito')
        ax1.set_ylabel('Real')
        ax1.set_title(f'Matriz de Confusão - {model_name}\n(Valores Absolutos)')
        
        # Matriz normalizada
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
        ax2.set_xlabel('Predito')
        ax2.set_ylabel('Real')
        ax2.set_title(f'Matriz de Confusão - {model_name}\n(Normalizada)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_bias_analysis(self, bias_results, bias_type='skin_type'):
        """Plota análise de viés com contexto brasileiro"""
        categories = list(bias_results.keys())
        accuracies = [bias_results[cat]['accuracy'] for cat in categories]
        sample_counts = [bias_results[cat]['sample_count'] for cat in categories]
        descriptions = [bias_results[cat]['description'] for cat in categories]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico 1: Acurácia por categoria
        bars1 = ax1.bar(range(len(categories)), accuracies, alpha=0.7, 
                        color=['lightcoral', 'lightblue', 'lightgreen'][:len(categories)])
        ax1.set_xlabel('Categorias')
        ax1.set_ylabel('Acurácia')
        ax1.set_title(f'Análise de Viés - {bias_type.replace("_", " ").title()}')
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(descriptions, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Gráfico 2: Distribuição de amostras
        bars2 = ax2.bar(range(len(categories)), sample_counts, alpha=0.7,
                        color=['orange', 'purple', 'brown'][:len(categories)])
        ax2.set_xlabel('Categorias')
        ax2.set_ylabel('Número de Amostras')
        ax2.set_title('Distribuição de Amostras por Categoria')
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(descriptions, rotation=45, ha='right')
        
        for bar, count in zip(bars2, sample_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_counts)*0.01,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_ablation_results(self, ablation_results):
        """Plota resultados dos estudos de ablação"""
        components = []
        accuracies = []
        
        if 'visual_only' in ablation_results:
            components.append('Visual Only')
            accuracies.append(ablation_results['visual_only']['accuracy'])
        
        if 'multimodal' in ablation_results:
            components.append('Multimodal')
            accuracies.append(ablation_results['multimodal']['accuracy'])
            
        if 'with_clahe' in ablation_results:
            components.append('Com CLAHE')
            accuracies.append(ablation_results['with_clahe']['accuracy'])
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(components, accuracies, alpha=0.7, 
                      color=['skyblue', 'lightcoral', 'lightgreen'][:len(components)])
        
        plt.xlabel('Componentes do Sistema')
        plt.ylabel('Acurácia')
        plt.title('Estudo de Ablação - Contribuição dos Componentes')
        plt.ylim(0, 1)
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Adicionar linha de melhoria multimodal se disponível
        if 'multimodal_improvement' in ablation_results:
            improvement = ablation_results['multimodal_improvement']
            plt.axhline(y=accuracies[0] + improvement, color='red', linestyle='--', 
                       alpha=0.7, label=f'Melhoria Multimodal: +{improvement:.3f}')
            plt.legend()
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# =============================================================================
# 8. PIPELINE COMPLETO DE EXPERIMENTOS
# =============================================================================

class DermAIExperimentPipeline:
    def __init__(self, data_path="./data", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.data_path = Path(data_path)
        self.device = device
        self.results = {}
        
        print(f"🚀 Inicializando pipeline de experimentos DermAI")
        print(f"💻 Dispositivo: {device}")
        print(f"📁 Caminho dos dados: {data_path}")
    
    def load_isic_data(self):
        """Carrega dados ISIC organizados por classe"""
        print("\n📊 Carregando dados ISIC...")
        
        train_images, train_labels = [], []
        val_images, val_labels = [], []
        test_images, test_labels = [], []
        
        # Mapeamento reverso para converter nomes de classe para índices
        reverse_mapping = {v: k for k, v in CLASS_MAPPING.items()}
        
        for split in ['train', 'val', 'test']:
            split_path = self.data_path / split
            if not split_path.exists():
                print(f"⚠️ Diretório {split_path} não encontrado. Usando dados simulados.")
                # Gerar dados simulados para demonstração
                images, labels = self._generate_synthetic_data(1000 if split == 'train' else 200)
            else:
                images, labels = self._load_split_data(split_path, reverse_mapping)
            
            if split == 'train':
                train_images, train_labels = images, labels
            elif split == 'val':
                val_images, val_labels = images, labels
            else:
                test_images, test_labels = images, labels
        
        print(f"✅ Dados carregados:")
        print(f"   Treino: {len(train_images)} imagens")
        print(f"   Validação: {len(val_images)} imagens") 
        print(f"   Teste: {len(test_images)} imagens")
        
        return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
    
    def _load_split_data(self, split_path, reverse_mapping):
        """Carrega dados de um split específico"""
        images, labels = [], []
        
        for class_folder in split_path.iterdir():
            if class_folder.is_dir() and class_folder.name in reverse_mapping:
                class_idx = list(CLASS_MAPPING.keys()).index(reverse_mapping[class_folder.name])
                
                for img_file in class_folder.glob("*.jpg"):
                    images.append(str(img_file))
                    labels.append(class_idx)
        
        return images, labels
    
    def _generate_synthetic_data(self, n_samples):
        """Gera dados sintéticos para demonstração"""
        images = [f"synthetic_img_{i}.jpg" for i in range(n_samples)]
        labels = np.random.randint(0, len(CLASS_MAPPING), n_samples).tolist()
        return images, labels
    
    def run_benchmark_experiments(self, train_data, val_data, test_data):
        """Executa benchmark completo de modelos"""
        print("\n" + "="*60)
        print("🏆 BENCHMARK DE MODELOS")
        print("="*60)
        
        train_images, train_labels = train_data
        val_images, val_labels = val_data
        test_images, test_labels = test_data
        
        # Criar datasets
        train_dataset = SkinLesionDataset(train_images, train_labels, transform_train)
        val_dataset = SkinLesionDataset(val_images, val_labels, transform_test)
        test_dataset = SkinLesionDataset(test_images, test_labels, transform_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        trainer = DermAIModelTrainer(self.device)
        benchmark_results = {}
        
        # 1. Modelo principal do projeto (NeuronZero/SkinCancerClassifier)
        print("🎯 Testando modelo principal (ProjectSkinClassifier)...")
        try:
            project_model = ProjectSkinClassifier(self.device)
            project_result = trainer.evaluate_model(project_model, test_loader, "ProjectSkinClassifier")
            benchmark_results['ProjectSkinClassifier'] = project_result
        except Exception as e:
            print(f"❌ Erro ao testar ProjectSkinClassifier: {e}")
        
        # 2. CNNs clássicas
        print("🔍 Testando CNNs clássicas...")
        cnn_models = ClassicalCNNs().get_all_models()
        
        for name, model in cnn_models.items():
            try:
                print(f"   📊 Treinando {name}...")
                train_result = trainer.train_model(model, train_loader, val_loader, name, epochs=3)
                test_result = trainer.evaluate_model(model, test_loader, name)
                benchmark_results[name] = test_result
            except Exception as e:
                print(f"❌ Erro ao treinar {name}: {e}")
        
        # 3. Sistema multimodal
        print("🤖 Testando sistema multimodal...")
        try:
            multimodal_model = MultimodalDermAI(self.device)
            mm_result = trainer.evaluate_model(multimodal_model, test_loader, "MultimodalDermAI")
            benchmark_results['MultimodalDermAI'] = mm_result
        except Exception as e:
            print(f"❌ Erro ao testar MultimodalDermAI: {e}")
        
        self.results['benchmark'] = benchmark_results
        return benchmark_results
    
    def run_robustness_tests(self, test_data, model=None):
        """Executa testes de robustez para condições amazônicas"""
        print("\n" + "="*60)
        print("🌳 TESTES DE ROBUSTEZ AMAZÔNICA")
        print("="*60)
        
        if model is None:
            try:
                model = ProjectSkinClassifier(self.device)
            except:
                print("❌ Erro ao carregar modelo para testes de robustez")
                return {}
        
        test_images, test_labels = test_data
        robustness_tester = AmazonianRobustnessTests(model, self.device)
        robustness_results = robustness_tester.test_amazonian_conditions(test_images, test_labels)
        
        self.results['robustness'] = robustness_results
        return robustness_results
    
    def run_bias_analysis(self, test_data_by_category, model=None):
        """Executa análise de viés"""
        print("\n" + "="*60)
        print("⚖️ ANÁLISE DE VIÉS")
        print("="*60)
        
        if model is None:
            try:
                model = ProjectSkinClassifier(self.device)
            except:
                print("❌ Erro ao carregar modelo para análise de viés")
                return {}
        
        bias_analyzer = BrazilianBiasAnalysis()
        bias_results = {}
        
        # Análise por tipo de pele (se dados disponíveis)
        if 'skin_types' in test_data_by_category:
            bias_results['skin_types'] = bias_analyzer.analyze_skin_type_bias(
                model, test_data_by_category['skin_types']
            )
        
        # Análise por região (se dados disponíveis)
        if 'regions' in test_data_by_category:
            bias_results['regions'] = bias_analyzer.analyze_regional_bias(
                model, test_data_by_category['regions']
            )
        
        self.results['bias'] = bias_results
        return bias_results
    
    def run_ablation_studies(self, train_data, val_data, test_data):
        """Executa estudos de ablação"""
        print("\n" + "="*60)
        print("🔬 ESTUDOS DE ABLAÇÃO")
        print("="*60)
        
        train_images, train_labels = train_data
        val_images, val_labels = val_data
        test_images, test_labels = test_data
        
        # Criar datasets
        train_dataset = SkinLesionDataset(train_images, train_labels, transform_train)
        val_dataset = SkinLesionDataset(val_images, val_labels, transform_test)
        test_dataset = SkinLesionDataset(test_images, test_labels, transform_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        ablation = DermAIAblationStudy(self.device)
        ablation_results = {}
        
        try:
            # Teste visual apenas
            visual_result = ablation.test_visual_only(train_loader, val_loader, test_loader)
            ablation_results.update(visual_result)
            
            # Teste multimodal vs visual
            comparison_result = ablation.test_multimodal_vs_visual(train_loader, val_loader, test_loader)
            ablation_results.update(comparison_result)
            
        except Exception as e:
            print(f"❌ Erro nos estudos de ablação: {e}")
        
        self.results['ablation'] = ablation_results
        return ablation_results
    
    def run_cross_validation(self, all_images, all_labels, k_folds=5):
        """Executa validação cruzada"""
        print("\n" + "="*60)
        print(f"📊 VALIDAÇÃO CRUZADA {k_folds}-FOLD")
        print("="*60)
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
            print(f"\n🔄 Fold {fold + 1}/{k_folds}")
            
            try:
                # Dividir dados
                train_images = [all_images[i] for i in train_idx]
                train_labels = [all_labels[i] for i in train_idx]
                val_images = [all_images[i] for i in val_idx]
                val_labels = [all_labels[i] for i in val_idx]
                
                # Criar datasets
                train_dataset = SkinLesionDataset(train_images, train_labels, transform_train)
                val_dataset = SkinLesionDataset(val_images, val_labels, transform_test)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Treinar modelo
                model = ProjectSkinClassifier(self.device)
                trainer = DermAIModelTrainer(self.device)
                
                result = trainer.train_model(model, train_loader, val_loader, f"Fold_{fold+1}", epochs=3)
                fold_results.append(result['best_accuracy'])
                
            except Exception as e:
                print(f"❌ Erro no fold {fold+1}: {e}")
                fold_results.append(0.0)
        
        cv_results = {
            'mean_accuracy': np.mean(fold_results),
            'std_accuracy': np.std(fold_results),
            'fold_accuracies': fold_results,
            'confidence_interval': np.percentile(fold_results, [2.5, 97.5])
        }
        
        self.results['cross_validation'] = cv_results
        return cv_results
    
    def visualize_all_results(self):
        """Gera todas as visualizações dos resultados"""
        print("\n" + "="*60)
        print("📈 VISUALIZAÇÃO DOS RESULTADOS")
        print("="*60)
        
        visualizer = DermAIResultsVisualizer()
        
        # Benchmark de modelos
        if 'benchmark' in self.results:
            visualizer.plot_model_comparison(self.results['benchmark'])
        
        # Robustez amazônica
        if 'robustness' in self.results:
            visualizer.plot_amazonian_robustness(self.results['robustness'])
        
        # Análise de viés
        if 'bias' in self.results:
            for bias_type, bias_data in self.results['bias'].items():
                visualizer.plot_bias_analysis(bias_data, bias_type)
        
        # Ablation studies
        if 'ablation' in self.results:
            visualizer.plot_ablation_results(self.results['ablation'])
        
        # Matrizes de confusão para modelo principal
        if 'benchmark' in self.results and 'ProjectSkinClassifier' in self.results['benchmark']:
            result = self.results['benchmark']['ProjectSkinClassifier']
            visualizer.plot_confusion_matrix_detailed(
                result['true_labels'], 
                result['predictions'], 
                'ProjectSkinClassifier'
            )
    
    def generate_final_report(self):
        """Gera relatório final dos experimentos"""
        print("\n" + "="*60)
        print("📋 RELATÓRIO FINAL DOS EXPERIMENTOS")
        print("="*60)
        
        # Benchmark
        if 'benchmark' in self.results:
            print("\n🏆 RESULTADOS DO BENCHMARK:")
            print("-" * 50)
            for model_name, result in self.results['benchmark'].items():
                print(f"{model_name:25} | Acc: {result['accuracy']:.3f} | F1: {result['f1_score']:.3f} | Recall: {result['recall']:.3f}")
        
        # Validação cruzada
        if 'cross_validation' in self.results:
            cv_result = self.results['cross_validation']
            print(f"\n📊 VALIDAÇÃO CRUZADA:")
            print(f"Acurácia Média: {cv_result['mean_accuracy']:.3f} ± {cv_result['std_accuracy']:.3f}")
            print(f"Intervalo de Confiança (95%): [{cv_result['confidence_interval'][0]:.3f}, {cv_result['confidence_interval'][1]:.3f}]")
        
        # Robustez
        if 'robustness' in self.results:
            print(f"\n🌳 ROBUSTEZ AMAZÔNICA:")
            print("-" * 50)
            for condition, result in self.results['robustness'].items():
                print(f"{condition:20} | Acc: {result['accuracy']:.1f}% | Retenção: {result['retention_rate']:.1f}%")
        
        # Análise de viés
        if 'bias' in self.results:
            print(f"\n⚖️ ANÁLISE DE VIÉS:")
            print("-" * 50)
            for bias_type, bias_data in self.results['bias'].items():
                print(f"\n{bias_type.upper()}:")
                for category, result in bias_data.items():
                    print(f"  {result['description']:25} | Acc: {result['accuracy']:.3f}")
        
        # Ablation
        if 'ablation' in self.results:
            print(f"\n🔬 ESTUDOS DE ABLAÇÃO:")
            print("-" * 50)
            ablation = self.results['ablation']
            if 'visual_only' in ablation:
                print(f"Visual Apenas: {ablation['visual_only']['accuracy']:.3f}")
            if 'multimodal' in ablation:
                print(f"Multimodal: {ablation['multimodal']['accuracy']:.3f}")
                if 'multimodal_improvement' in ablation:
                    print(f"Melhoria Multimodal: +{ablation['multimodal_improvement']:.3f} ({ablation['improvement_percentage']:.1f}%)")
        
        print("\n✅ Experimentos DermAI concluídos com sucesso!")
        
        return self.results

# =============================================================================
# 9. EXEMPLO DE USO COMPLETO
# =============================================================================

def run_dermia_experiments():
    """Executa pipeline completo de experimentos para o projeto DermAI"""
    
    # Inicializar pipeline
    pipeline = DermAIExperimentPipeline(data_path="./data")
    
    try:
        # 1. Carregar dados
        train_data, val_data, test_data = pipeline.load_isic_data()
        
        # 2. Benchmark de modelos
        benchmark_results = pipeline.run_benchmark_experiments(train_data, val_data, test_data)
        
        # 3. Testes de robustez
        robustness_results = pipeline.run_robustness_tests(test_data)
        
        # 4. Validação cruzada
        all_images = train_data[0] + val_data[0] + test_data[0]
        all_labels = train_data[1] + val_data[1] + test_data[1]
        cv_results = pipeline.run_cross_validation(all_images, all_labels, k_folds=3)
        
        # 5. Estudos de ablação
        ablation_results = pipeline.run_ablation_studies(train_data, val_data, test_data)
        
        # 6. Análise de viés (dados simulados para exemplo)
        # Em uso real, você forneceria dados categorizados por tipo de pele/região
        test_data_by_category = {
            'skin_types': {
                'tipo_I_II': (test_data[0][:50], test_data[1][:50]),
                'tipo_III_IV': (test_data[0][50:150], test_data[1][50:150]),
                'tipo_V_VI': (test_data[0][150:200], test_data[1][150:200])
            }
        }
        bias_results = pipeline.run_bias_analysis(test_data_by_category)
        
        # 7. Visualizar resultados
        pipeline.visualize_all_results()
        
        # 8. Gerar relatório final
        final_results = pipeline.generate_final_report()
        
        # 9. Salvar resultados
        results_path = Path("experiment_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            # Converter arrays numpy para listas para serialização JSON
            serializable_results = {}
            for key, value in final_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for k, v in value.items():
                        if hasattr(v, 'tolist'):
                            serializable_results[key][k] = v.tolist()
                        else:
                            serializable_results[key][k] = v
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados salvos em: {results_path}")
        
        return final_results
        
    except Exception as e:
        print(f"❌ Erro durante execução dos experimentos: {e}")
        import traceback
        traceback.print_exc()
        return {}

# Executar experimentos
if __name__ == "__main__":
    print("🚀 Iniciando experimentos DermAI...")
    print("📋 Para executar, descomente a linha abaixo:")
    print("# results = run_dermia_experiments()")
    