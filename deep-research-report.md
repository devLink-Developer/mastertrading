# Trading de futuros en criptomonedas principales con Smart Money Concepts y diseño de bots

## Panorama del mercado de derivados cripto y selección de activos

El “trading de futuros” en cripto, en la práctica, está dominado por **perpetuos** (perps): derivados sin vencimiento que replican la exposición al subyacente mediante **apalancamiento** y un mecanismo de **funding** para mantener el precio del contrato cerca del spot. Este diseño se ha convertido en el núcleo de la formación de precio y de la liquidez en muchas criptomonedas, hasta el punto de que los agregadores de mercado reportan que, a escala global, el volumen de derivados puede representar una parte mayoritaria del volumen total cripto en determinados periodos. citeturn6search25turn6search20turn13search8

En cuanto a qué activos “priorizar”, conviene separar dos ideas:  
- **Top por capitalización** (lo que el usuario suele llamar “top 10”).  
- **Top por liquidez real en derivados** (lo que más importa para futuros: spreads, profundidad de libro, volumen, open interest y estabilidad operativa).

Como referencia temporal verificable, un “snapshot” histórico de entity["company","CoinMarketCap","crypto data aggregator"] (2 de febrero de 2026) lista como top diez por capitalización: **BTC, ETH, USDT, BNB, XRP, USDC, SOL, TRX, DOGE y BCH**. citeturn3view0turn3view1turn3view5turn3view7  
Dos matices importantes para futuros:

1) **USDT y USDC son stablecoins**: rara vez son el “subyacente” de futuros direccionales (porque su objetivo es 1:1), pero son **cruciales** como **colateral, moneda de margen y denominación** en futuros lineales (por ejemplo, “USDT-margined”). citeturn3view1turn3view5turn1view2  
2) En cripto coexisten **plataformas cripto-nativas** y **mercados regulados tradicionales**. En estos últimos, por ejemplo, entity["organization","CME Group","derivatives exchange operator"] publica tamaños de contrato y horas de negociación para futuros cripto (incluyendo BTC y también tamaños para ETH, SOL y XRP en su material de criptoderivados). citeturn7view1 Además, bolsas tradicionales en Asia han anunciado listados de perpetuos cripto para perfiles institucionales, señal de integración del producto “perp” en infraestructura clásica. citeturn6news44

Por último, una advertencia estructural: **futuros + apalancamiento** implica que movimientos relativamente pequeños del precio pueden tener impacto grande sobre el margen. La entity["organization","Commodity Futures Trading Commission","us derivatives regulator"] (EE. UU.) advierte explícitamente que operar futuros vía cuentas apalancadas amplifica riesgos y puede implicar pérdidas que excedan la inversión inicial. citeturn12view1 A la vez, reguladores como la entity["organization","Financial Conduct Authority","uk financial regulator"] han mantenido prohibiciones para venta de cripto-derivados a minoristas en el Reino Unido, lo que ilustra que el acceso puede depender de jurisdicción y categoría de cliente. citeturn9search3turn9search15

## Cómo funcionan los futuros y perpetuos en cripto

El punto diferencial frente a futuros tradicionales es que los **perpetuos no vencen**, por lo que no existe el “tirón” mecánico de convergencia al spot que sí caracteriza a muchos futuros con fecha. En su lugar, el sistema usa **funding rate** y, en la práctica, una combinación de **index price** y **mark price** para gestión de riesgo. citeturn13search8turn11view0turn5search3turn5search11

**Funding (qué es y por qué mueve el mercado)**  
En un diseño típico de perps:
- Si el perp cotiza con prima vs spot, el funding tiende a ser **positivo** y **los largos pagan a los cortos**, incentivando ventas del perp (o coberturas) que empujan a la convergencia. Si cotiza con descuento, funding puede ser **negativo** y los cortos pagan a los largos. citeturn1view2turn13search8turn11view0  
- La literatura académica y de políticas públicas subraya que el funding es un **mecanismo central** del “anclaje” spot-perp. citeturn13search8turn11view0

En la práctica, un exchange grande como entity["company","Binance","crypto exchange"] documenta (i) que el funding son pagos periódicos entre largos y cortos, (ii) que el objetivo es anclar el precio del perpetuo al índice spot y (iii) una fórmula operativa habitual:  
\[
\text{Funding Amount}=\text{Nominal Value}\times \text{Funding Rate},
\quad \text{Nominal Value}=\text{Mark Price}\times \text{contract size}.
\]  
También especifica intervalos por defecto (p. ej., 00:00, 08:00 y 16:00 UTC) y la posibilidad de ajustar frecuencia en condiciones extremas. citeturn1view2turn13search3turn13search11

Un detalle microestructural importante para bots: existe evidencia de que hay **picos de actividad alrededor de las horas de funding** (lo que tiene sentido porque el coste/ingreso de carry se materializa en esos instantes). Un artículo académico sobre diseño de contrato y estructura de mercado en perps documenta picos de negociación en torno a las horas de funding (00:00, 08:00, 16:00 UTC) en el mercado perpetuo. citeturn13search32

**Mark price, index price y liquidaciones**  
En perps, para evitar que una última transacción aislada y con baja liquidez dispare liquidaciones “injustas”, muchas plataformas usan **mark price** para P&L no realizado, margen y liquidación. entity["company","Deribit","crypto derivatives exchange"] lo define como un precio de referencia para procesos de riesgo (P&L, margen y liquidaciones) cuyo objetivo es reflejar una estimación “robusta” incluso cuando el libro está temporalmente distorsionado. citeturn5search3  
La contraparte conceptual es el **index price**, que suele agregarse a partir de precios spot en varias plataformas y con controles de calidad. citeturn5search11

Las liquidaciones en cripto suelen ser **automáticas y rápidas**. Deribit describe un sistema de auto-liquidación incremental: cuando la cuenta no tiene equity suficiente para mantener posiciones (según su motor de riesgo), se cierra parte de la posición, sin “margin call” previo. citeturn5search7  
En periodos de estrés esto se traduce en **cascadas**: cierres forzosos → más movimiento adverso → más liquidaciones. Reuters ha reportado episodios recientes con miles de millones en liquidaciones asociadas a movimientos en bitcoin en un entorno de venta global de activos de riesgo. citeturn9news41turn10view0

**Liquidez, profundidad y por qué importa para futuros**  
La liquidez no es solo “volumen”; en ejecución importa la **profundidad cerca del precio**. Reuters (citando a un analista de entity["company","Kaiko","crypto market data provider"]) explica que una métrica habitual es la “profundidad al 1%”, es decir, cuánta cantidad puede cruzarse a ±1% del precio sin generar gran impacto; y reporta un descenso de esa profundidad promedio en bitcoin desde 2025 a comienzos de 2026, lo que vuelve el mercado más propenso a movimientos bruscos por órdenes relativamente pequeñas. citeturn10view0  
Esto es especialmente crítico en futuros porque el apalancamiento transforma esos movimientos bruscos en riesgo de liquidación.

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["crypto perpetual futures funding rate chart","bitcoin liquidation cascade chart","limit order book depth visualization","wyckoff accumulation distribution schematic"],"num_per_query":1}

## Smart Money Concepts en criptomonedas

“Smart Money Concepts” (SMC) no es un estándar académico único; es un **conjunto de heurísticas de lectura de precio** (estructura, liquidez, “bloques de órdenes”, imbalances) popularizado en comunidades de trading. Para convertirlo en algo útil en cripto (y especialmente en futuros), lo más productivo es **reformular SMC como hipótesis microestructurales**: si el precio se mueve por (i) búsqueda de liquidez, (ii) reasignación de inventario de creadores de mercado y (iii) shocks de información, entonces ciertos patrones del gráfico pueden ser señales imperfectas de esos procesos. La literatura de microestructura destaca que variables como desequilibrio de flujo de órdenes, spreads, profundidad y patrones de llegada de trades explican parte de la variación de retornos a horizontes muy cortos. citeturn4search10turn4search4

**De Wyckoff a SMC (marco para “acumulación/distribución”)**  
Mucho SMC en cripto se parece a un “Wyckoff operativo”: detectar fases (acumulación/markup/distribución/markdown) y eventos (springs, upthrust, etc.). Binance Academy presenta los esquemas de acumulación y distribución como una parte popular de Wyckoff dentro de la comunidad cripto. citeturn4search17 Un resumen general de Wyckoff, incluyendo las cuatro fases, puede encontrarse en entity["organization","Investopedia","financial education site"]. citeturn4search3

**Bloques de órdenes, liquidez y “gaps” en clave cripto**  
En SMC, “order block” suele referirse a la zona del último impulso previo a un desplazamiento fuerte (la idea es que ahí quedó interés institucional). Una explicación típica en el universo ICT/SMC contrasta order blocks con fair value gaps (FVG) como “impulsos con desequilibrio”. citeturn4search32turn4search20  
Un punto clave: estos conceptos son **cartográficos** (se infieren del precio) y no equivalen automáticamente a una huella verificable de órdenes limitadas. Para acercarlos a una realidad cuantitativa puedes:
- Tratar un “FVG” como una proxy de **movimiento con baja permanencia** (precio atravesando niveles con poco intercambio), lo que conecta con episodios de **libros finos** o discontinuidades de ejecución. citeturn4search16turn10view0  
- Exigir confirmación con **datos de derivados**: funding, open interest, volatilidad realizada, y medidas de liquidez.

**Qué cambia en cripto frente a Forex o acciones**  
1) **Mercado 24/7, pero con “sesiones” reales**. Aunque no hay apertura/cierre oficial, existen patrones intradiarios robustos. Un estudio académico sobre “tea time” encuentra patrones pronunciados de actividad intradía: retornos, volumen, volatilidad e iliquidez varían sistemáticamente (en UTC) y muestran máximos en ventanas concretas del día. citeturn4search15turn4search11  
2) **Fragmentación**. La formación de precio ocurre en múltiples venues (spot y derivados). Un trabajo reciente sobre integridad de libro en entornos fragmentados argumenta que los movimientos de precio sin ejecución (o con ejecución limitada) son más prevalentes en mercados fragmentados, con implicaciones directas para stops, slippage y modelos de ejecución. citeturn4search18  
3) **Derivados como motor**. Parte de la evidencia empírica sugiere que el perp “tether-margined” en un gran exchange puede ser una fuente dominante de transmisión de volatilidad hacia otros instrumentos. citeturn13search28 Esto encaja con una lectura SMC adaptada: en cripto, “liquidity sweeps” no solo cazan stops; también recorren zonas donde hay **liquidaciones** y where margin gets stressed.

## Patrones típicos: sesiones, fin de semana, macro y correlaciones

En cripto, muchos “movimientos típicos” no son mágicos: emergen de **calendarios (macro)**, **sesiones de liquidez**, **mecánicas de perps (funding)** y **conexión con mercados tradicionales**.

**Sesiones y “aperturas” relevantes**  
- Un paper de alta frecuencia en arXiv (datos 2020–2022) identifica fases de actividad alineadas con sesiones asiática/europea/estadounidense y detecta aumentos recurrentes de actividad cerca de “full hours”, además de coincidencias con horarios de publicaciones macro en EE. UU. citeturn4search19  
- En derivados, existe evidencia de que ciertos picos intradiarios se alinean con eventos de mercado tradicional. Un estudio sobre opciones de bitcoin encuentra que parte de la actividad se concentra en horas que coinciden con la apertura de la bolsa de entity["city","Nueva York","new york, ny, us"] y que ese pico se atenúa en fin de semana, sugiriendo spillovers desde renta variable. citeturn9search13  
- En derivados regulados, el “open” es literal: por ejemplo, CME reporta horarios de negociación casi continuos de domingo a viernes (no 24/7), lo que puede crear transiciones de liquidez entre “semana tradicional” y “fin de semana cripto”. citeturn7view1

**Fin de semana: rango, liquidez y “gaps” de ejecución**  
El fin de semana suele comportarse distinto porque cambia la composición de participantes y la infraestructura fiat. Evidencias relevantes:
- Kaiko documenta que la **cuota de BTC negociada en fines de semana** cayó de forma marcada entre 2018 y 2023 (del 24% al 17%), interpretándolo como deterioro de condiciones de liquidez en fin de semana y cambios en participación institucional/infraestructura. citeturn9search25  
- Un estudio de microestructura sobre “liquidity commonality” encuentra estacionalidad semanal en spreads, volumen y volatilidad, con una commonality que tiende a **decaer en fin de semana**. citeturn9search1turn4search1  

Desde una óptica SMC cuantitativa, el “rango de fin de semana” puede modelarse como: menor profundidad → mayor impacto de órdenes → probabilidad mayor de barridas de liquidez y de movimientos discontinuos, algo coherente con el argumento de Reuters sobre profundidad al 1% y movimientos más erráticos cuando la liquidez se contrae. citeturn10view0

**Noticias económicas y releases: por qué “mueven” cripto**  
Aunque cripto tenga narrativa propia, hay evidencia consistente de sensibilidad a anuncios macro:
- Un estudio (2020) analiza si anuncios del FOMC y otros datos macro afectan precios de bitcoin. citeturn0search2  
- Un trabajo con datos de 5 minutos (2016–2023) encuentra que la volatilidad de BTC/ETH responde significativamente a categorías seleccionadas de noticias macro, con un rol destacado de noticias de política monetaria de EE. UU., incluso con efectos en periodos pre-anuncio. citeturn4search2  
- La idea de “sorpresa informativa” está reforzada por investigación de la entity["organization","Reserva Federal","us central bank"] sobre cómo la atención del inversor influye en la incorporación de anuncios macro (como CPI y NFP) en precios de mercado. citeturn0search26  

Operativamente, un bot de futuros que ignore el calendario macro tenderá a infravalorar colas de volatilidad y slippage en ventanas de release.

**Relación con bolsa tradicional y descorrelación**  
La correlación cripto–acciones ha sido **regímen-dependiente** y, en periodos de estrés, tiende a aumentar. Reuters reportó que bitcoin se ha vuelto más sensible a señales de política monetaria y a movimientos en renta variable/tecnología, con incrementos en correlación media en 2025 frente a 2024 según datos de mercado (LSEG) en su cobertura. citeturn5news45turn10view0  
A nivel académico, un estudio (2018–2025) documenta aumento de correlación tras hitos de adopción institucional (p. ej., ETP/ETF y mecanismos corporativos), con picos reportados y dependencia del régimen. citeturn12view2  
En paralelo, modelos macro-financieros encuentran spillovers: shocks en cripto pueden transmitirse a mercados financieros globales, y viceversa, bajo ciertos regímenes. citeturn5search1turn0search6

La “descorrelación” no debe asumirse como un estado permanente; es más útil modelarla como un **indicador dinámico** (correlación rodante, DCC, coherencia wavelet, etc.) y como una **condición de régimen** para ajustar el riesgo, no como una promesa de diversificación.

## Datos y mediciones: precios, funding, open interest y liquidez

Para operar futuros con SMC “cuantificable” necesitas distinguir entre *datos para señal* y *datos para riesgo/ejecución*.

**Datos mínimos (por mercado y por activo)**  
- **Precio**: last, mid, best bid/ask, OHLCV.  
- **Referencia de riesgo**: index price y mark price (si el exchange lo publica). citeturn5search3turn5search11turn1view2  
- **Derivados**: funding rate (actual e histórico), frecuencia de funding, open interest, tasa de liquidaciones (si existe feed), basis (futuro vs spot). citeturn1view2turn13search8turn6search19  
- **Microestructura**: spread, profundidad por niveles, imbalance del libro, flujo de trades. La literatura reciente resume estos factores como explicativos de dinámica de precio a muy corto plazo. citeturn4search10turn4search4  

**Cómo obtener valores actuales de forma robusta**  
En la práctica hay tres capas:

1) **API nativa del exchange (REST/WebSocket)**: suele ser la mejor para tiempo real y precisión de derivados (mark price, funding, libro). Por ejemplo, Binance documenta que usuarios de API pueden consultar datos de funding de futuros (p. ej. endpoint `GET /fapi/v1/fundingInfo`). citeturn1view2turn6search17  
2) **Librerías de abstracción**: la documentación de CCXT explica el enfoque de API unificada para acceso a múltiples exchanges (útil para prototipos, menos ideal para HFT). citeturn6search2turn6search6turn6search18  
3) **Agregadores**: entity["company","CoinGecko","crypto data aggregator"] documenta endpoints tanto de precio “simple” como históricos, y también ofrece endpoints específicos de **derivatives exchanges** (incluyendo campos como open interest). citeturn6search7turn6search11turn6search19  

**Regla operativa crítica para bots de futuros**: para riesgo (y, a menudo, para triggers de liquidación) debes trabajar con **mark price**, no con último precio. Esto es coherente con cómo Deribit define el mark price como referencia de P&L, margen y liquidaciones. citeturn5search3turn5search7

**Cómo “leer” SMC con datos cuantitativos (puente práctica–medición)**  
Una forma disciplinada de adaptar SMC es convertir cada idea en una variable medible:

- “**Liquidez arriba/abajo**” → clusters de máximos/mínimos + *book depth* (p. ej., profundidad al 0,5%/1%) + spreads. Esto conecta con medidas de profundidad usadas en investigación/cobertura de mercado. citeturn10view0turn4search4  
- “**Barrida de liquidez (sweep)**” → ruptura de un extremo reciente + rápida reversión + salto en volumen/trades + empeoramiento temporal del spread o de profundidad. Los patrones intradía de actividad y volatilidad están documentados en cripto. citeturn4search15turn4search19  
- “**Confirmación**” → cambio de estructura (BOS/CHoCH) + estabilización del funding (evitar entrar cuando el coste de carry es extremo) + señales de normalización de liquidez. Existe evidencia de que el funding es pieza clave de anclaje y puede ser persistente o volátil según régimen. citeturn13search8turn13search35turn1view2  

## Diseño matemático y arquitectura de un bot de futuros

Un bot coherente con SMC y con microestructura debe tratarse más como un **sistema de gestión de riesgo + ejecución** que como un “predictor” puro. La evidencia empírica en cripto sugiere que la liquidez varía por hora y día, y que eventos macro y mecánicas de derivados (funding, liquidaciones) alteran el entorno de trading. citeturn4search15turn9search25turn13search32turn4search2turn5search7

### Estado matemático mínimo del sistema

Define, por instrumento \(i\):

- Precio de referencia \(M_i(t)\): **mark price**. citeturn5search3turn1view2  
- Posición \(q_i(t)\) (positiva largo, negativa corto).  
- Precio medio de entrada \(p^{\text{entry}}_i\).  
- Equity de cuenta \(E(t)\) (balance + P&L no realizado − comisiones − funding acumulado).  
- Exposición nocional \(N_i(t)=|q_i(t)|\cdot M_i(t)\).  
- Restricción de margen: liquidación si \(E(t)\) cae bajo un umbral ligado a mantenimiento; el mecanismo exacto depende del exchange, pero el concepto de liquidación automática por insuficiencia de equity es estándar y está documentado por Deribit. citeturn5search7turn5search3  

**Funding esperado**  
Si el exchange publica funding actual \(f_i(t)\) y su próxima hora de cobro/pago, el coste/ingreso esperado para una ventana \(\Delta t\) se aproxima como:
\[
\text{Funding}(t,\Delta t)\approx N_i(t)\cdot f_i(t)
\]
en la convención del exchange (en Binance se expresa explícitamente como nominal \(\times\) funding rate). citeturn1view2turn13search3

### Arquitectura recomendada del bot

**Capa de adquisición (market data)**  
- WebSocket para trades y libro (si el exchange lo permite).  
- REST para snapshots y para redundancia.  
- Fuentes externas para agregación y checks (CoinGecko) cuando tenga sentido. citeturn6search7turn6search19turn6search2  

**Capa de features (microestructura + derivados + calendario)**  
- Liquidez: spread, profundidad, imbalance del libro; hay evidencia de patrones intradía y de que la variación del libro impacta P&L y costes de ejecución. citeturn4search4turn4search15  
- Derivados: funding, open interest (si disponible), distancias a precio de liquidación estimado (según reglas del exchange), volatilidad realizada. citeturn1view2turn6search19turn5search3  
- Calendario: marca ventanas de macro-noticias (especialmente policy/EE. UU.) porque la volatilidad reacciona y puede hacerlo antes del anuncio. citeturn4search2turn0search2turn4search19  

**Capa de decisión (señal SMC cuantificada)**  
Aquí conviene no “misticar” SMC: úsalo como *template* de condiciones.

### Algoritmos de largos y cortos basados en SMC con filtros cuantitativos

A continuación, dos plantillas (no garantías de rentabilidad; son estructuras para backtesting riguroso). La razón para incluir filtros de funding/horario es que hay evidencia de concentración de actividad alrededor de funding y de variación intradía de liquidez. citeturn13search32turn4search15turn1view2turn9search25

**Algoritmo tipo largo (reversión tras barrida de liquidez + confirmación)**  
1) **Contexto**: detectar rango o tramo bajista extendido (HTF).  
2) **Evento SMC**: sweep por debajo de un mínimo relevante (ej., mínimo de sesión o de X velas) con rechazo rápido (cierre por encima del nivel barrido).  
3) **Confirmación de estructura**: CHoCH/BOS alcista en marco menor (romper el último “lower high” local).  
4) **Filtro de derivados**: evitar entrar en la zona de mayor coste de carry si el funding es fuertemente positivo (crowded longs). Si el funding viene negativo y empieza a normalizarse, es coherente con un mercado donde el perp estaba con descuento o el sesgo era bajista. La literatura enfatiza que funding es el mecanismo clave del anclaje y que suele ser positivo en promedio (largos pagando), por lo que extremos pueden reflejar posicionamiento. citeturn11view0turn13search8turn13search35turn1view2  
5) **Filtro de liquidez**: no ejecutar si el spread y la profundidad empeoran (evitar horas de iliquidez marcada; hay patrones intradía documentados). citeturn4search15turn10view0  
6) **Entrada**: pullback a zona “order block/FVG” *solo* si además el libro muestra normalización (mejor profundidad, menor spread).  
7) **Stop**: por debajo del mínimo del sweep (más un colchón basado en volatilidad intradía).  
8) **Gestión**: reducir exposición antes de anuncios macro de alta sensibilidad (monetaria/EE. UU.) o escalar riesgo a la baja; hay evidencia de que estos anuncios mueven volatilidad. citeturn4search2turn0search2

**Algoritmo tipo corto (continuación bajista con “liquidity grab” arriba)**  
1) **Contexto**: tendencia bajista o distribución (HTF), compatible con Wyckoff/SMC. citeturn4search17turn4search3  
2) **Evento**: sweep por encima de un máximo local (toma de stops) y giro con vela impulsiva bajista.  
3) **Confirmación**: BOS bajista en marco menor.  
4) **Derivados**: si el funding está muy negativo, el mercado puede estar “crowded short”; esto no invalida el corto, pero exige tamaños menores y stops más estrictos (riesgo de short squeeze). La mecánica funding-long/short y su función de anclaje están documentadas. citeturn1view2turn13search8turn11view0  
5) **Horario**: extremar cautela en fin de semana (menor participación y señales de peor liquidez) y alrededor de funding times (picos de actividad). citeturn9search25turn13search32turn13search3  
6) **Entrada**: retesteo a zona de oferta (order block) con ejecución escalonada si la profundidad lo permite.  

### Estacionalidad: cómo medirla y cómo incorporarla al bot

“Estacionalidad” (day-of-week, fin de semana, intradía) debe tratarse como un **bloque de modelos**:

- Estudios clásicos encuentran “day-of-week effects” en cripto (p. ej., anomalías específicas de ciertos activos/periodos) y trabajos recientes siguen explorando anomalías de calendario y momentum en cripto, lo que justifica testear por activo y por régimen. citeturn0search7turn9search9  
- En microestructura, hay evidencia de estacionalidad semanal en spreads/volumen/volatilidad y de deterioro relativo en fin de semana. citeturn9search1turn9search25  
- Intradía, los patrones de actividad se alinean con ventanas horarias globales, y la literatura documenta picos alrededor de la “tea time” del entity["city","Londres","london, england, uk"] (en UTC) y coincidencias con releases macro. citeturn4search15turn4search19turn4search11  

Implementación práctica en el bot:  
- Construye un **modelo de volatilidad/liquidez por hora y por día** (matriz \(7 \times 24\)) por activo.  
- Ajusta el tamaño de posición y/o el umbral de señal según esa matriz (p. ej. reducir riesgo donde spreads y slippage esperados suben). Esto encaja con evidencia de que costes dependen de liquidez y de patrones intradía. citeturn4search4turn10view0  

### Controles de riesgo imprescindibles

1) **Límite de pérdida diaria/semanal y kill-switch**: la CFTC advierte que el apalancamiento puede llevar a pérdidas mayores que la inversión inicial. citeturn12view1  
2) **Cap de apalancamiento efectivo**: medir apalancamiento como \(\sum_i N_i(t)/E(t)\), no como el “slider” del exchange.  
3) **Modelo de slippage**: usar profundidad del libro y spreads; la investigación sobre liquidez intradía en exchanges cripto muestra patrones aprovechables para minimizar costes de trading. citeturn4search4  
4) **Gestión de funding como coste de carry**: incorporar expectativa de funding en el expected value de la operación; es el coste/ingreso que mantiene el anclaje spot–perp. citeturn1view2turn13search8turn11view0  
5) **Regímenes macro/correlación**: en “risk-off” la correlación con acciones puede subir; Reuters y estudios académicos describen integración creciente y dependencia del régimen. citeturn10view0turn12view2turn5search1  

En síntesis: para que SMC sea útil en futuros cripto a nivel profesional, conviene (i) definirlo como un conjunto de hipótesis verificables sobre liquidez y estructura, (ii) validarlo con variables específicas de perps (mark price, funding, open interest, liquidaciones) y (iii) integrarlo en un bot en el que el “alpha” está subordinado a ejecución y control de riesgo en un mercado 24/7 con estacionalidad intradía y sensibilidad macro demostrable. citeturn5search3turn1view2turn4search15turn4search2turn10view0