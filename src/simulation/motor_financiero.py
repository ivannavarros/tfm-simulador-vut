# ============================================================
# motor_financiero.py
# Motor de cálculo económico-financiero del simulador VUT
# ============================================================

import numpy as np
from dataclasses import dataclass
from typing import Dict
from scipy.optimize import brentq


@dataclass
class ParametrosInversion:
    """Parámetros de entrada del inversor"""

    # Inversión inicial
    precio_compra: float
    gastos_compraventa_pct: float
    reforma: float
    mobiliario: float
    licencias: float

    # Financiación
    pct_financiado: float
    tipo_interes: float
    plazo_anos: int

    # Ingresos
    precio_noche: float
    ocupacion: float
    noches_disponibles: int
    crecimiento_ingresos: float

    # Costes operativos
    coste_limpieza_rotacion: float
    rotaciones_año: int
    mantenimiento_anual: float
    comunidad_anual: float
    seguro_anual: float
    ibi_anual: float
    suministros_anual: float
    comision_plataforma_pct: float
    gestion_externa_pct: float

    # Fiscalidad
    tipo_irpf: float

    # Análisis
    tasa_descuento: float
    horizonte_anos: int


def calcular_cuota_hipoteca(capital: float, tipo_anual: float, plazo_anos: int) -> float:
    """Sistema francés: cuota mensual constante"""
    if tipo_anual == 0 or capital == 0:
        return 0.0
    tipo_mensual = tipo_anual / 12
    n = plazo_anos * 12
    cuota = capital * (tipo_mensual * (1 + tipo_mensual)**n) / ((1 + tipo_mensual)**n - 1)
    return cuota


def calcular_motor(params: ParametrosInversion, escenario: str = 'neutro') -> Dict:
    """Calcula todos los indicadores financieros para un escenario dado."""

    factores = {
        'optimista': {'ingresos': 1.20, 'costes': 0.90, 'ocupacion': 1.15},
        'neutro':    {'ingresos': 1.00, 'costes': 1.00, 'ocupacion': 1.00},
        'pesimista': {'ingresos': 0.80, 'costes': 1.15, 'ocupacion': 0.80},
    }
    f = factores[escenario]

    # Inversión inicial
    gastos_compraventa = params.precio_compra * params.gastos_compraventa_pct
    inversion_total    = (params.precio_compra + gastos_compraventa +
                          params.reforma + params.mobiliario + params.licencias)
    capital_ajeno  = params.precio_compra * params.pct_financiado
    capital_propio = inversion_total - capital_ajeno

    # Financiación
    cuota_mensual = calcular_cuota_hipoteca(capital_ajeno, params.tipo_interes, params.plazo_anos)
    cuota_anual   = cuota_mensual * 12

    # Proyección anual
    flujos_caja = []
    for año in range(1, params.horizonte_anos + 1):

        ocupacion_ajustada = min(params.ocupacion * f['ocupacion'], 1.0)
        ingresos_brutos = (params.precio_noche *
                           ocupacion_ajustada *
                           params.noches_disponibles *
                           f['ingresos'] *
                           (1 + params.crecimiento_ingresos) ** (año - 1))

        coste_limpieza  = params.coste_limpieza_rotacion * params.rotaciones_año * f['costes']
        comision_plat   = ingresos_brutos * params.comision_plataforma_pct
        gestion_externa = ingresos_brutos * params.gestion_externa_pct

        costes_fijos = ((params.mantenimiento_anual +
                         params.comunidad_anual +
                         params.seguro_anual +
                         params.ibi_anual +
                         params.suministros_anual) * f['costes'])

        costes_financieros = cuota_anual if año <= params.plazo_anos else 0
        costes_operativos  = coste_limpieza + comision_plat + gestion_externa + costes_fijos

        base_imponible = max(ingresos_brutos - costes_operativos - costes_financieros, 0)
        impuestos      = base_imponible * params.tipo_irpf

        fcn = ingresos_brutos - costes_operativos - costes_financieros - impuestos

        flujos_caja.append({
            'año':                año,
            'ingresos_brutos':    round(ingresos_brutos, 2),
            'costes_operativos':  round(costes_operativos, 2),
            'costes_financieros': round(costes_financieros, 2),
            'impuestos':          round(impuestos, 2),
            'fcn':                round(fcn, 2),
        })

    fcn_serie = np.array([x['fcn'] for x in flujos_caja])

    # VAN
    van = -inversion_total + sum(fc / (1 + params.tasa_descuento)**t
                                  for t, fc in enumerate(fcn_serie, 1))

    # TIR
    def npv(tasa, flujos, inversion):
        return -inversion + sum(f / (1 + tasa)**t for t, f in enumerate(flujos, 1))

    tir = None
    try:
        if npv(-0.99, fcn_serie, inversion_total) * npv(10.0, fcn_serie, inversion_total) < 0:
            tir = brentq(npv, -0.99, 10.0, args=(fcn_serie, inversion_total))
    except Exception:
        tir = None

    # Payback simple (sobre capital propio)
    acumulado = 0
    payback = None
    for t, fc in enumerate(fcn_serie, 1):
        acumulado += fc
        if acumulado >= capital_propio:
            payback = t
            break

    # Payback descontado (sobre capital propio)
    acumulado_desc = 0
    payback_desc = None
    for t, fc in enumerate(fcn_serie, 1):
        acumulado_desc += fc / (1 + params.tasa_descuento)**t
        if acumulado_desc >= capital_propio:
            payback_desc = t
            break

    # Rentabilidades
    ingresos_año1      = flujos_caja[0]['ingresos_brutos']
    rentabilidad_bruta = ingresos_año1 / params.precio_compra
    rentabilidad_neta  = flujos_caja[0]['fcn'] / inversion_total
    roe                = flujos_caja[0]['fcn'] / capital_propio if capital_propio > 0 else 0

    # Punto muerto
    costes_totales_año1    = (flujos_caja[0]['costes_operativos'] +
                              flujos_caja[0]['costes_financieros'])
    punto_muerto_ocupacion = (costes_totales_año1 /
                              (params.precio_noche * params.noches_disponibles)
                              if params.precio_noche > 0 else None)

    return {
        'escenario':              escenario,
        'inversion_total':        round(inversion_total, 2),
        'capital_propio':         round(capital_propio, 2),
        'capital_ajeno':          round(capital_ajeno, 2),
        'cuota_mensual':          round(cuota_mensual, 2),
        'van':                    round(van, 2),
        'tir':                    round(tir, 4) if tir is not None else None,
        'payback':                payback,
        'payback_descontado':     payback_desc,
        'rentabilidad_bruta':     round(rentabilidad_bruta, 4),
        'rentabilidad_neta':      round(rentabilidad_neta, 4),
        'roe':                    round(roe, 4),
        'punto_muerto_ocupacion': round(punto_muerto_ocupacion, 4) if punto_muerto_ocupacion else None,
        'flujos_caja':            flujos_caja,
    }


def calcular_tres_escenarios(params: ParametrosInversion) -> Dict:
    """Calcula los tres escenarios simultáneamente"""
    return {
        'optimista': calcular_motor(params, 'optimista'),
        'neutro':    calcular_motor(params, 'neutro'),
        'pesimista': calcular_motor(params, 'pesimista'),
    }


if __name__ == '__main__':

    params = ParametrosInversion(
        precio_compra=150000,
        gastos_compraventa_pct=0.10,
        reforma=5000,
        mobiliario=4000,
        licencias=800,
        pct_financiado=0.0,
        tipo_interes=0.035,
        plazo_anos=20,
        precio_noche=120,
        ocupacion=0.65,
        noches_disponibles=330,
        crecimiento_ingresos=0.02,
        coste_limpieza_rotacion=50,
        rotaciones_año=90,
        mantenimiento_anual=800,
        comunidad_anual=1200,
        seguro_anual=350,
        ibi_anual=400,
        suministros_anual=600,
        comision_plataforma_pct=0.12,
        gestion_externa_pct=0.0,
        tipo_irpf=0.24,
        tasa_descuento=0.06,
        horizonte_anos=10,
    )

    resultados = calcular_tres_escenarios(params)

    for escenario, res in resultados.items():
        print(f"\n{'='*40}")
        print(f"ESCENARIO: {escenario.upper()}")
        print(f"{'='*40}")
        print(f"Inversión total:      {res['inversion_total']:>10,.0f}€")
        print(f"Capital propio:       {res['capital_propio']:>10,.0f}€")
        print(f"Cuota mensual:        {res['cuota_mensual']:>10,.0f}€")
        print(f"VAN:                  {res['van']:>10,.0f}€")
        if res['tir'] is not None:
            print(f"TIR:                  {res['tir']*100:>9.1f}%")
        else:
            print(f"TIR:                   no calculable")
        print(f"Payback:              {str(res['payback']):>10} años")
        print(f"Payback descontado:   {str(res['payback_descontado']):>10} años")
        print(f"Rentabilidad bruta:   {res['rentabilidad_bruta']*100:>9.1f}%")
        print(f"Rentabilidad neta:    {res['rentabilidad_neta']*100:>9.1f}%")
        print(f"ROE:                  {res['roe']*100:>9.1f}%")
        if res['punto_muerto_ocupacion']:
            print(f"Punto muerto ocup.:   {res['punto_muerto_ocupacion']*100:>9.1f}%")