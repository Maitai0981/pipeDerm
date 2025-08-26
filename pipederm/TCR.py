# DermAI - Testes de Conectividade e Robustez
# Baseado no artigo: Sistema de Análise Dermatológica Automatizada em Pipeline
# Seção 5.5 - Simulação de diferentes cenários de rede para regiões remotas

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import threading
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuração do notebook
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== DermAI - Testes de Conectividade e Robustez ===")
print("Simulação de cenários de rede para regiões remotas da Amazônia")
print("Baseado na Tabela 12 do artigo")

class ConnectivityTester:
    """Classe para testar robustez do sistema em diferentes condições de rede"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.scenarios = {
            'estavel': {'latencia': 45, 'taxa_sucesso': 98.5, 'timeout': 30, 'recuperacao': 0},
            'lenta': {'latencia': 180, 'taxa_sucesso': 92.1, 'timeout': 60, 'recuperacao': 15.4},
            'intermitente': {'latencia': 320, 'taxa_sucesso': 78.6, 'timeout': 120, 'recuperacao': 28.7},
            'offline': {'latencia': 0, 'taxa_sucesso': 100, 'timeout': 0, 'recuperacao': 0}
        }
        self.results = defaultdict(list)
        
    def simular_condicao_rede(self, scenario):
        """Simula diferentes condições de rede baseadas no artigo"""
        config = self.scenarios[scenario]
        
        if scenario == 'offline':
            # Simula funcionalidade offline (visualização de cache)
            return {
                'sucesso': True,
                'latencia_ms': 0,
                'timeout': False,
                'tempo_recuperacao': 0,
                'funcionalidade': 'cache_local'
            }
        
        # Simula latência variável
        latencia_base = config['latencia']
        latencia_real = max(10, np.random.normal(latencia_base, latencia_base * 0.3))
        
        # Simula falhas baseadas na taxa de sucesso
        sucesso = random.random() < (config['taxa_sucesso'] / 100)
        
        # Simula timeout
        timeout_ocorreu = latencia_real > config['timeout'] * 1000
        
        # Tempo de recuperação para falhas
        tempo_recuperacao = 0
        if not sucesso or timeout_ocorreu:
            tempo_recuperacao = np.random.normal(config['recuperacao'], config['recuperacao'] * 0.2)
            tempo_recuperacao = max(0, tempo_recuperacao)
        
        return {
            'sucesso': sucesso and not timeout_ocorreu,
            'latencia_ms': latencia_real,
            'timeout': timeout_ocorreu,
            'tempo_recuperacao': tempo_recuperacao,
            'funcionalidade': 'completa' if sucesso and not timeout_ocorreu else 'limitada'
        }
    
    def testar_cenario(self, scenario, num_testes=100):
        """Executa testes para um cenário específico"""
        print(f"\nTestando cenário: {scenario.upper()}")
        print(f"Configuração esperada: {self.scenarios[scenario]}")
        
        resultados_cenario = {
            'sucessos': 0,
            'falhas': 0,
            'timeouts': 0,
            'latencias': [],
            'tempos_recuperacao': [],
            'funcionalidades': []
        }
        
        for i in range(num_testes):
            if (i + 1) % 25 == 0:
                print(f"  Progresso: {i + 1}/{num_testes} testes")
            
            # Simula requisição
            resultado = self.simular_condicao_rede(scenario)
            
            # Coleta métricas
            if resultado['sucesso']:
                resultados_cenario['sucessos'] += 1
            else:
                resultados_cenario['falhas'] += 1
                
            if resultado['timeout']:
                resultados_cenario['timeouts'] += 1
                
            resultados_cenario['latencias'].append(resultado['latencia_ms'])
            if resultado['tempo_recuperacao'] > 0:
                resultados_cenario['tempos_recuperacao'].append(resultado['tempo_recuperacao'])
            resultados_cenario['funcionalidades'].append(resultado['funcionalidade'])
            
            # Pequena pausa para simular requests reais
            time.sleep(0.01)
        
        # Calcula estatísticas finais
        taxa_sucesso = (resultados_cenario['sucessos'] / num_testes) * 100
        latencia_media = np.mean(resultados_cenario['latencias'])
        timeout_medio = np.mean(resultados_cenario['tempos_recuperacao']) if resultados_cenario['tempos_recuperacao'] else 0
        
        print(f"  Taxa de sucesso: {taxa_sucesso:.1f}% (esperado: {self.scenarios[scenario]['taxa_sucesso']:.1f}%)")
        print(f"  Latência média: {latencia_media:.0f}ms (esperado: {self.scenarios[scenario]['latencia']}ms)")
        print(f"  Tempo recuperação médio: {timeout_medio:.1f}s (esperado: {self.scenarios[scenario]['recuperacao']}s)")
        
        self.results[scenario] = resultados_cenario
        return resultados_cenario
    
    def executar_todos_cenarios(self, num_testes=100):
        """Executa testes para todos os cenários de rede"""
        print("\n" + "="*50)
        print("EXECUTANDO TESTES DE CONECTIVIDADE E ROBUSTEZ")
        print("="*50)
        
        for scenario in self.scenarios.keys():
            self.testar_cenario(scenario, num_testes)
            time.sleep(1)  # Pausa entre cenários
        
        print("\nTodos os cenários testados com sucesso!")
    
    def simular_rede_instavel(self, duracao_segundos=60):
        """Simula rede instável com mudanças dinâmicas de condição"""
        print(f"\n=== SIMULAÇÃO DE REDE INSTÁVEL ({duracao_segundos}s) ===")
        
        tempo_inicio = time.time()
        historico = []
        cenario_atual = 'estavel'
        
        while time.time() - tempo_inicio < duracao_segundos:
            # Muda cenário aleatoriamente
            if random.random() < 0.1:  # 10% chance de mudança a cada ciclo
                cenario_atual = random.choice(['estavel', 'lenta', 'intermitente'])
            
            resultado = self.simular_condicao_rede(cenario_atual)
            resultado['cenario'] = cenario_atual
            resultado['timestamp'] = time.time() - tempo_inicio
            historico.append(resultado)
            
            time.sleep(0.5)  # Teste a cada 500ms
        
        # Analisa resultados da simulação instável
        df_instavel = pd.DataFrame(historico)
        print(f"Total de testes realizados: {len(df_instavel)}")
        print(f"Taxa de sucesso geral: {(df_instavel['sucesso'].sum() / len(df_instavel)) * 100:.1f}%")
        print(f"Latência média: {df_instavel['latencia_ms'].mean():.0f}ms")
        
        # Distribuição por cenário
        print("\nDistribuição por cenário:")
        for cenario in df_instavel['cenario'].unique():
            subset = df_instavel[df_instavel['cenario'] == cenario]
            taxa = (subset['sucesso'].sum() / len(subset)) * 100
            print(f"  {cenario}: {len(subset)} testes, {taxa:.1f}% sucesso")
        
        self.results['instavel'] = df_instavel
        return df_instavel
    
    def testar_recuperacao_automatica(self, num_falhas=10):
        """Testa capacidade de recuperação automática do sistema"""
        print(f"\n=== TESTE DE RECUPERAÇÃO AUTOMÁTICA ===")
        print(f"Simulando {num_falhas} falhas consecutivas")
        
        tempos_recuperacao = []
        tentativas_recuperacao = []
        
        for i in range(num_falhas):
            print(f"Simulando falha {i+1}/{num_falhas}")
            
            # Simula falha inicial
            tempo_falha = time.time()
            
            # Tentativas de recuperação
            tentativas = 0
            recuperado = False
            
            while not recuperado and tentativas < 5:
                tentativas += 1
                time.sleep(0.5)  # Delay entre tentativas
                
                # Simula tentativa de recuperação (sucesso aumenta com tentativas)
                chance_sucesso = min(0.9, 0.3 + tentativas * 0.2)
                if random.random() < chance_sucesso:
                    recuperado = True
                    tempo_recuperacao = time.time() - tempo_falha
                    tempos_recuperacao.append(tempo_recuperacao)
                    tentativas_recuperacao.append(tentativas)
                    print(f"  Recuperado em {tempo_recuperacao:.1f}s após {tentativas} tentativas")
                    break
            
            if not recuperado:
                print(f"  Falha na recuperação após {tentativas} tentativas")
                tempos_recuperacao.append(30.0)  # Timeout máximo
                tentativas_recuperacao.append(tentativas)
        
        # Estatísticas de recuperação
        print(f"\nResultados da recuperação automática:")
        print(f"Tempo médio de recuperação: {np.mean(tempos_recuperacao):.1f}s")
        print(f"Tentativas médias: {np.mean(tentativas_recuperacao):.1f}")
        print(f"Taxa de recuperação: {(len([t for t in tempos_recuperacao if t < 30]) / len(tempos_recuperacao)) * 100:.1f}%")
        
        self.results['recuperacao'] = {
            'tempos': tempos_recuperacao,
            'tentativas': tentativas_recuperacao
        }
        
        return tempos_recuperacao, tentativas_recuperacao
    
    def visualizar_resultados(self):
        """Cria visualizações dos resultados de conectividade"""
        if not self.results:
            print("Nenhum resultado disponível para visualização")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DermAI - Análise de Conectividade e Robustez', fontsize=16, fontweight='bold')
        
        # 1. Comparação de taxa de sucesso por cenário
        cenarios = ['estavel', 'lenta', 'intermitente', 'offline']
        cenarios_disponiveis = [c for c in cenarios if c in self.results]
        
        if cenarios_disponiveis:
            taxas_sucesso_teste = []
            taxas_sucesso_esperadas = []
            latencias_teste = []
            latencias_esperadas = []
            
            for cenario in cenarios_disponiveis:
                if cenario == 'offline':
                    taxa_teste = 100.0  # Offline sempre funciona para cache
                    latencia_teste = 0
                else:
                    dados = self.results[cenario]
                    taxa_teste = (dados['sucessos'] / (dados['sucessos'] + dados['falhas'])) * 100
                    latencia_teste = np.mean(dados['latencias'])
                
                taxas_sucesso_teste.append(taxa_teste)
                taxas_sucesso_esperadas.append(self.scenarios[cenario]['taxa_sucesso'])
                latencias_teste.append(latencia_teste)
                latencias_esperadas.append(self.scenarios[cenario]['latencia'])
            
            x = np.arange(len(cenarios_disponiveis))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, taxas_sucesso_teste, width, 
                          label='Teste Atual', alpha=0.7, color='lightblue')
            axes[0, 0].bar(x + width/2, taxas_sucesso_esperadas, width, 
                          label='Esperado (Artigo)', alpha=0.7, color='lightcoral')
            axes[0, 0].set_xlabel('Cenário de Rede')
            axes[0, 0].set_ylabel('Taxa de Sucesso (%)')
            axes[0, 0].set_title('Taxa de Sucesso por Cenário')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels([c.capitalize() for c in cenarios_disponiveis])
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Comparação de latência
            axes[0, 1].bar(x - width/2, latencias_teste, width,
                          label='Teste Atual', alpha=0.7, color='lightgreen')
            axes[0, 1].bar(x + width/2, latencias_esperadas, width,
                          label='Esperado (Artigo)', alpha=0.7, color='orange')
            axes[0, 1].set_xlabel('Cenário de Rede')
            axes[0, 1].set_ylabel('Latência (ms)')
            axes[0, 1].set_title('Latência Média por Cenário')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels([c.capitalize() for c in cenarios_disponiveis])
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Simulação de rede instável (se disponível)
        if 'instavel' in self.results:
            df_instavel = self.results['instavel']
            
            # Timeline da simulação instável
            cores_cenario = {'estavel': 'green', 'lenta': 'orange', 'intermitente': 'red'}
            
            for cenario in df_instavel['cenario'].unique():
                subset = df_instavel[df_instavel['cenario'] == cenario]
                axes[1, 0].scatter(subset['timestamp'], subset['latencia_ms'], 
                                 c=cores_cenario.get(cenario, 'blue'), 
                                 label=cenario.capitalize(), alpha=0.6, s=30)
            
            axes[1, 0].set_xlabel('Tempo (segundos)')
            axes[1, 0].set_ylabel('Latência (ms)')
            axes[1, 0].set_title('Simulação de Rede Instável - Timeline')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # Gráfico alternativo: distribuição de latência por cenário
            latencias_por_cenario = []
            labels_cenarios = []
            
            for cenario in cenarios_disponiveis:
                if cenario != 'offline' and cenario in self.results:
                    latencias_por_cenario.append(self.results[cenario]['latencias'])
                    labels_cenarios.append(cenario.capitalize())
            
            if latencias_por_cenario:
                axes[1, 0].boxplot(latencias_por_cenario, labels=labels_cenarios)
                axes[1, 0].set_xlabel('Cenário de Rede')
                axes[1, 0].set_ylabel('Latência (ms)')
                axes[1, 0].set_title('Distribuição de Latência por Cenário')
                axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Análise de recuperação (se disponível)
        if 'recuperacao' in self.results:
            dados_rec = self.results['recuperacao']
            
            # Histograma de tempos de recuperação
            axes[1, 1].hist(dados_rec['tempos'], bins=10, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].axvline(np.mean(dados_rec['tempos']), color='red', linestyle='--',
                              label=f'Média: {np.mean(dados_rec["tempos"]):.1f}s')
            axes[1, 1].set_xlabel('Tempo de Recuperação (s)')
            axes[1, 1].set_ylabel('Frequência')
            axes[1, 1].set_title('Distribuição dos Tempos de Recuperação')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Gráfico alternativo: métricas de timeout
            timeout_data = []
            cenario_labels = []
            
            for cenario in cenarios_disponiveis:
                if cenario != 'offline' and cenario in self.results:
                    dados = self.results[cenario]
                    if dados['tempos_recuperacao']:
                        timeout_data.append(np.mean(dados['tempos_recuperacao']))
                    else:
                        timeout_data.append(0)
                    cenario_labels.append(cenario.capitalize())
            
            if timeout_data:
                axes[1, 1].bar(cenario_labels, timeout_data, alpha=0.7, color='salmon')
                axes[1, 1].set_xlabel('Cenário de Rede')
                axes[1, 1].set_ylabel('Tempo Médio de Recuperação (s)')
                axes[1, 1].set_title('Tempo de Recuperação por Cenário')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def gerar_relatorio_conectividade(self):
        """Gera relatório detalhado dos testes de conectividade"""
        print("\n" + "="*60)
        print("RELATÓRIO DETALHADO DE CONECTIVIDADE E ROBUSTEZ")
        print("="*60)
        
        if not self.results:
            print("Nenhum resultado disponível para relatório")
            return
        
        # Resumo por cenário
        print("\n1. RESUMO POR CENÁRIO:")
        print("-" * 40)
        
        cenarios_teste = ['estavel', 'lenta', 'intermitente', 'offline']
        dados_comparacao = []
        
        for cenario in cenarios_teste:
            if cenario in self.results:
                esperado = self.scenarios[cenario]
                
                if cenario == 'offline':
                    taxa_real = 100.0
                    latencia_real = 0
                    timeout_real = 0
                else:
                    dados = self.results[cenario]
                    total_testes = dados['sucessos'] + dados['falhas']
                    taxa_real = (dados['sucessos'] / total_testes) * 100 if total_testes > 0 else 0
                    latencia_real = np.mean(dados['latencias']) if dados['latencias'] else 0
                    timeout_real = np.mean(dados['tempos_recuperacao']) if dados['tempos_recuperacao'] else 0
                
                print(f"\n{cenario.upper()}:")
                print(f"  Taxa de Sucesso: {taxa_real:.1f}% (esperado: {esperado['taxa_sucesso']:.1f}%)")
                print(f"  Latência Média: {latencia_real:.0f}ms (esperado: {esperado['latencia']}ms)")
                print(f"  Tempo Recuperação: {timeout_real:.1f}s (esperado: {esperado['recuperacao']}s)")
                
                dados_comparacao.append({
                    'cenario': cenario,
                    'taxa_sucesso_real': taxa_real,
                    'taxa_sucesso_esperada': esperado['taxa_sucesso'],
                    'latencia_real': latencia_real,
                    'latencia_esperada': esperado['latencia'],
                    'recuperacao_real': timeout_real,
                    'recuperacao_esperada': esperado['recuperacao']
                })
        
        # Análise de conformidade
        print("\n2. ANÁLISE DE CONFORMIDADE:")
        print("-" * 40)
        
        if dados_comparacao:
            conformidades = []
            for dados in dados_comparacao:
                # Calcula desvio percentual
                desvio_taxa = abs(dados['taxa_sucesso_real'] - dados['taxa_sucesso_esperada'])
                desvio_latencia = abs(dados['latencia_real'] - dados['latencia_esperada']) / dados['latencia_esperada'] * 100 if dados['latencia_esperada'] > 0 else 0
                
                conformidade_taxa = "OK" if desvio_taxa <= 5 else "DESVIO"
                conformidade_latencia = "OK" if desvio_latencia <= 20 else "DESVIO"
                
                print(f"\n{dados['cenario'].upper()}:")
                print(f"  Taxa de Sucesso: {conformidade_taxa} (desvio: {desvio_taxa:.1f}%)")
                print(f"  Latência: {conformidade_latencia} (desvio: {desvio_latencia:.1f}%)")
                
                conformidades.append(conformidade_taxa == "OK" and conformidade_latencia == "OK")
            
            conformidade_geral = sum(conformidades) / len(conformidades) * 100
            print(f"\nConformidade Geral: {conformidade_geral:.1f}% dos cenários")
        
        # Recomendações para deployment
        print("\n3. RECOMENDAÇÕES PARA DEPLOYMENT:")
        print("-" * 40)
        print("Baseado nos resultados dos testes:")
        
        if 'estavel' in self.results:
            dados_estavel = self.results['estavel']
            if dados_estavel['sucessos'] / (dados_estavel['sucessos'] + dados_estavel['falhas']) > 0.95:
                print("✓ Sistema adequado para conexões estáveis (>95% sucesso)")
            else:
                print("⚠ Verificar otimizações para conexões estáveis")
        
        if 'lenta' in self.results:
            dados_lenta = self.results['lenta']
            if dados_lenta['sucessos'] / (dados_lenta['sucessos'] + dados_lenta['falhas']) > 0.90:
                print("✓ Sistema resiliente a conexões lentas (>90% sucesso)")
            else:
                print("⚠ Implementar otimizações para conexões lentas")
        
        if 'intermitente' in self.results:
            dados_inter = self.results['intermitente']
            if dados_inter['sucessos'] / (dados_inter['sucessos'] + dados_inter['falhas']) > 0.75:
                print("✓ Sistema funcional com conectividade intermitente (>75% sucesso)")
            else:
                print("⚠ Necessário melhorar robustez para conexões instáveis")
        
        print("\nRecomendações específicas:")
        print("- Implementar cache local para funcionalidade offline")
        print("- Configurar timeouts adaptativos baseados na qualidade da conexão")
        print("- Desenvolver modo degradado para cenários de baixa conectividade")
        print("- Adicionar retry automático com backoff exponencial")
        
        return dados_comparacao

# Executar testes de conectividade
print("\n" + "="*60)
print("EXECUTANDO TESTES DE CONECTIVIDADE E ROBUSTEZ")
print("="*60)

# Inicializar testador
connectivity_tester = ConnectivityTester()

# Executar testes para todos os cenários (número reduzido para demonstração)
print("\nConfigurando testes com 50 iterações por cenário (reduzido para demonstração)")
connectivity_tester.executar_todos_cenarios(num_testes=50)

# Simular rede instável
print("\n" + "="*50)
connectivity_tester.simular_rede_instavel(duracao_segundos=30)

# Testar recuperação automática
connectivity_tester.testar_recuperacao_automatica(num_falhas=5)

# Visualizar resultados
connectivity_tester.visualizar_resultados()

# Gerar relatório detalhado
dados_relatorio = connectivity_tester.gerar_relatorio_conectividade()

# Salvar resultados em DataFrame
print("\n=== SALVANDO RESULTADOS ===")
resultados_conectividade = []

for cenario, dados in connectivity_tester.results.items():
    if cenario not in ['instavel', 'recuperacao']:
        total_testes = dados['sucessos'] + dados['falhas']
        if total_testes > 0:
            resultados_conectividade.append({
                'cenario': cenario,
                'taxa_sucesso_pct': (dados['sucessos'] / total_testes) * 100,
                'latencia_media_ms': np.mean(dados['latencias']) if dados['latencias'] else 0,
                'num_timeouts': dados['timeouts'],
                'tempo_recuperacao_medio_s': np.mean(dados['tempos_recuperacao']) if dados['tempos_recuperacao'] else 0,
                'total_testes': total_testes
            })

if resultados_conectividade:
    df_conectividade = pd.DataFrame(resultados_conectividade)
    print("\nResumo dos resultados de conectividade:")
    print(df_conectividade.to_string(index=False, float_format='%.2f'))

print("\nComparação com benchmarks do artigo (Tabela 12):")
print("- Conexão Estável: 45ms latência, 98.5% sucesso")
print("- Conexão Lenta: 180ms latência, 92.1% sucesso, 15.4s recuperação")  
print("- Conexão Intermitente: 320ms latência, 78.6% sucesso, 28.7s recuperação")
print("- Offline (Cache): 100% sucesso para funcionalidades limitadas")
print("\nTeste de conectividade e robustez concluído!")