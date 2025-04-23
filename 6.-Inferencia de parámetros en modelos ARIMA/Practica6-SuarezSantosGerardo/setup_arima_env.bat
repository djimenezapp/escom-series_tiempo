@echo off
echo ========================================
echo  CREANDO ENTORNO VIRTUAL "arima-env"
echo ========================================
py -3.10 -m venv arima-env

echo ========================================
echo  ACTIVANDO ENTORNO
echo ========================================
call arima-env\Scripts\activate

echo ========================================
echo  INSTALANDO DEPENDENCIAS
echo ========================================
pip install numpy==1.24.4 pandas matplotlib scikit-learn statsmodels pmdarima

echo ========================================
echo  EJECUTANDO main.py
echo ========================================
python main.py

pause

