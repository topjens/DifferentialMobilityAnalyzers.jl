# +
# This file contains the basic DMAconfig functions (viscosity,transfer function etc.)
#
# Author: Markus Petters (mdpetter@ncsu.edu)
# 	      Department of Marine Earth and Atmospheric Sciences
#         NC State University
#         Raleigh, NC 27605
#
#         May, 2018
#-


"""
    clean(x)
    
Defined as shorthand:

```julia    
clean(x) = map(x -> x < 0.0 ? 0.0 : x, x)
```

The function removes negative numbers and set them zero. It is used to cleanup 
inverted size distribution data, which may contain small negative values from
inversion noise. 
"""
clean(x) = map(x -> x < 0.0 ? 0.0 : x, x)

"""
    Σ(f, i)
    
Defined as shorthand:

```julia    
Σ(f, i) = mapreduce(f, +, 1:i)
```

The function evaluates f(X) for X = [1,...,i] and sums the result. If f(X)
evaluates to a vector or SizeDistribution, the sum is the sum of the vectors or 
SizeDistributions.

Example Usage
```julia
Tc = getTc(Λ)
Σ(k -> Tc(k,100.0),2)  # evaluate the sum of Tc(1, 100.0), Tc(2, 100.0)
```
"""
Σ(f, i) = mapreduce(f, +, 1:i)


@doc raw"""
    λ(Λ:DMAconfig, ::CarrierGas)

λ is the mean free path of air in [m]. It  depends on temperature [K] and Pressure [Pa]. 
Temperature and pressure are taken from the DMA configuration. Currently dry air 
and N2 are supported.

``\lambda = 6.6 \times 10^{-8}\frac{101315}{p}\frac{t}{293.15}`` 

Example Usage

```julia
t,p = 295.15, 1e5                        
qsa,qsh = 1.66e-5, 8.3e-5                     
r₁,r₂,l = 9.37e-3,1.961e-2,0.44369               
Λ = DMAconfig(t,p,qsa,qsh,r₁,r₂,l,0.0,:-,6,:cylindrical,N2()) 
mfp = λ(Λ, N2())
```
"""
λ(Λ::DMAconfig, ::Air) = 6.6e-8 * (101315.0 ./ Λ.p) * (Λ.t / 293.15)
λ(Λ::DMAconfig, ::N2) = 5.9e-8 * (101315.0 ./ Λ.p) * (Λ.t / 293.15)

@doc raw"""
    η(Λ::DMAconfig, ::CarrierGas)

η is the viscosity of air in [Pa s] and depends on temperature [K]. Temperature 
is taken from the DMA configuration. Currently dry air and N2 is supported.

``\eta = 1.83245\times10^{-5} \exp \left(1.5 \ln \left[\frac{T}{296.1}\right]\right)\left 
(\frac{406.55}{T+110.4} \right)`` 

Example Usage

```julia
t,p = 295.15, 1e5                        
qsa,qsh = 1.66e-5, 8.3e-5                     
r₁,r₂,l = 9.37e-3,1.961e-2,0.44369               
Λ = DMAconfig(t,p,qsa,qsh,r₁,r₂,l,0.0,:-,6,:cylindrical,Air()) 
viscosity = η(Λ, Air())
```
"""
η(Λ::DMAconfig, ::Air) = 1.83245e-5 * exp(1.5 * log(Λ.t / 296.1)) * (406.55) / (Λ.t + 110.4)
η(Λ::DMAconfig, ::N2) = 1.663e-5 * (Λ.t / 273.0)^1.5 * (380.0) / (Λ.t + 107.0) 

@doc raw"""
    cc(Λ::DMAconfig, d)

Cunningham slip-flow correction factor. The slip flow correction accounts for the 
decreased drag particles experience relative to Stokes' drag force when particle 
size approaches the scale of the mean free path of air. It is computed following 
Hinds (1999) Eq. 3.20. Temperature and pressure are taken from the DMA configuration.
The units of diameter are in [m] and the function accepts scalars or arrays.

``
c_c = 1+\frac{\lambda}{d_p} \left(2.34+1.05 \exp \left[-0.39 \frac{d_p}{\lambda}\right]\right)
``

where ``d_p`` is the particle diameter and ``\lambda`` is the mean 
free path of air, which is computed as a function of pressure and temperature. 

Example Usage

```julia
Dp = exp10.(range(log10(1e-9), stop=log10(1000e-9), length=100))
t,p = 295.15, 1e5                        
qsa,qsh = 1.66e-5, 8.3e-5                    
r₁,r₂,l = 9.37e-3,1.961e-2,0.4436, Λ.gas9               
Λ = DMAconfig(t,p,qsa,qsh,r₁,r₂,l,0.0,:-,6,:cylindrical) 
correction = cc(Λ, Dp)
```
"""
cc(Λ::DMAconfig, d) = 1.0 .+ λ(Λ, Λ.gas) ./ d .* (2.34 .+ 1.05 .* exp.(-0.39 .* d ./ λ(Λ, Λ.gas)))

@doc raw"""
    dab(Λ::DMAconfig, d)

The diffusion coefficient of particles in air, ``d_{ab}``, describes the random 
displacement of particles due to Brownian motion. It is computed via the Stokes-Einstein 
relation (Hinds, 1999, Eq. 7.20). Temperature and pressure are taken from the DMA 
configuration. The units of diameter are in [m] and the function accepts scalars or arrays.

``d_{ab} = \frac{k_bTc_c}{3\pi\eta d_p}``

where ``k_b`` is Boltzmann's constant and ``\eta`` is the viscosity of air in [Pa s], 
``c_c`` is the Cunningham slip flow correction and ``d_p`` is the particle diameter.
``d_{ab}`` is in [m² s⁻¹].

Example Usage

```julia
Dp = exp10.(range(log10(1e-9), stop=log10(1000e-9), length=100))
t,p = 295.15, 1e5                        
qsa,qsh = 1.66e-5, 8.3e-5                     
r₁,r₂,l = 9.37e-3,1.961e-2,0.44369               
Λ = DMAconfig(t,p,qsa,qsh,r₁,r₂,l,0.0,:-,6,:cylindrical) 
diffusion_coefficient = dab(Λ,Dp)
```
"""
dab(Λ::DMAconfig, d) = kb * Λ.t * cc(Λ, d) ./ (3.0π * η(Λ, Λ.gas) * d)

u = (Λ, D) -> @. D * Λ.leff / Λ.qsa
Peff =
    u -> @. 0.82 * exp(-11.5u) +
       0.1 * exp(-70.0u) +
       0.03 * exp(-180.0u) +
       0.02 * exp(-340.0u)
Taf = (x, μ, σ) -> @. 0.5 * (1.0 + erf((x - μ) / (√2σ)))

@doc raw"""
    Tl(Λ::DMAconfig, Dp)

Penetration efficiency through the TSI cylindrical DMA using the parameterization by 
Reineking & Porstendörfer (1986). The particle diameter Dp is in [nm].
    
``T_l = 0.82\exp(-11.5u)+0.1\exp(-70.0u)+0.03\exp(-180.0u)+0.02\exp(-340.0u)``
    
where ``u = \frac{d_{ab} l_{eff}}{q_{sa}}$, $l_{eff}`` is the parameterized effective 
diffusion length, and ``q_{sa}`` is the aerosol flow rate through the DMA. 

!!! note

    Λ contains the effective length, aerosol flow rate, temperature and pressure to 
    compute ``d_{ab}``. To treat multiple DMAs with different {leff, qsa, 
    t, p} in a single script, the function Tl is  embedded  in the 
    DifferentialMobilityAnalyzer data type.

"""
function Tl(Λ::DMAconfig, Dp)
    if Λ.leff == 0.0
        return ones(length(Dp))
    else
        return clean(Peff(u(Λ, dab(Λ, Dp * 1e-9))))
    end
end

function Tl(Λ::DMAconfig, Z, k)
    if Λ.leff == 0.0
        return ones(length(Z))
    else
        Dp = map(zs -> ztod(Λ, k, zs) * 1e-9, Z)
        return clean(Peff(u(Λ, dab(Λ, Dp))))
    end
end

@doc raw"""
    getTc(Λ::DMAconfig)

Returns a function 
    
```julia 
Tc(k::Integer, Dp)
``` 
    
to compute the charging efficiency. Tc depends of the polarity set in DMAconfig.

Charging efficiency (charge equilibrium) obtained in the bipolar charger is computed based 
on the parameterized measurements by Wiedensohler et al. (1988) with coefficients taken 
from the TSI 3080 Manual (2009). 

``T_c(k) = 10^{\left\{ \sum_{i=1}^6 a_i (k) \left[ \ln \left(\frac{D_p}{nm}\right) \right]^{i-1} \right\}}``

where ``k = -2,-1,1,2`` is the number and polarity of particle charge and ``a_i`` are 
empirical coefficients. 

For ``k \ge \pm 3``, the formula from the TSI manual is used:

``T_c(k) = \frac{e}{\sqrt{4\pi^2\epsilon D_pk_bT}} \exp \left( \frac{-\frac{\left[|k| - 
2\pi\epsilon D_pk_bT \ln(0.875)\right]^2}{e^2}}{ \frac{4\pi\epsilon D_pk_bT}{e^2}} \right)``

where ``e`` is the elementary charge and ``\epsilon`` is the dielectric constant for air.

Example Usage
```julia
t,p = 295.15, 1e5                        
qsa,qsh = 1.66e-5, 8.3e-5                     
r₁,r₂,l = 9.37e-3,1.961e-2,0.44369               
Λ = DMAconfig(t,p,qsa,qsh,r₁,r₂,l,0.0,:-,6,:cylindrical) 
Tc = getTc(Λ)
Tc(1,100.0) # Note that Dp is in units of nm!
```

!!! note

    The function Tc is computed during DMA grid initialization and embedded in the 
    DifferentialMobilityAnalyzer data type. It is usually accessed through this grid.
    The diameter Dp is in units of nm.

"""
function getTc(Λ::DMAconfig)
    if Λ.polarity == :+    # positive polarity power supply
        A1 = convert(Array{Float64}, ax[!, :n1])
        A2 = convert(Array{Float64}, ax[!, :n2])
    elseif Λ.polarity == :-
        A1 = convert(Array{Float64}, ax[!, :p1])
        A2 = convert(Array{Float64}, ax[!, :p2])
    end

    fc = Function[]
    xi = d -> log10.(d)
    f1 =
        d ->
            10.0 .^ (
                A1[1] * xi(d) .^ 0.0 +
                A1[2] * xi(d) .^ 1.0 +
                A1[3] * xi(d) .^ 2.0 +
                A1[4] * xi(d) .^ 3.0 +
                A1[5] * xi(d) .^ 4.0 +
                A1[6] * xi(d) .^ 5.0
            )
    f2 =
        d ->
            10.0 .^ (
                A2[1] * xi(d) .^ 0.0 +
                A2[2] * xi(d) .^ 1.0 +
                A2[3] * xi(d) .^ 2.0 +
                A2[4] * xi(d) .^ 3.0 +
                A2[5] * xi(d) .^ 4.0 +
                A2[6] * xi(d) .^ 5.0
            )
    push!(fc, f1)
    push!(fc, f2)

    for i = 3:Λ.m
        fi = @. dp ->
            ec ./ (sqrt.(4.0 * π^2.0 * eps * dp * 1e-9 * kb * Λ.t)) .*
            exp.(
                -(
                    i * 1.0 - 2.0 * π * eps * dp * 1e-9 * kb * Λ.t * log(0.875) / ec .^ 2.0
                ) .^ 2.0 ./ (4.0 * π * eps * dp * 1e-9 * kb * Λ.t / ec^2.0),
            )
        push!(fc, fi)
    end
    Tc(k::Integer, Dp) = fc[k](Dp)
    return Tc
end

@doc raw"""
    dtoz(Λ::DMAconfig, d)

The function returns the mobility ``z`` according to  

``d_p =  \frac{kec_c}{3\pi \eta z^s}``

where ``e`` is the elementary charge,  ``k`` is the number of charges on the particle, 
``c_c`` is the Cunningham correction factor, and ``\eta`` is the viscosity of the fluid. 
The diameter in dtoz is in units of [m].

Example Usage
```julia
t,p = 295.15, 1e5                        
qsa,qsh = 1.66e-5, 8.3e-5                     
r₁,r₂,l = 9.37e-3,1.961e-2,0.44369               
Λ = DMAconfig(t,p,qsa,qsh,r₁,r₂,l,0.0,:-,6,:cylindrical) 
mobility = dtoz(Λ,dp*1e-9) # [m2 V-1 s-1]
```
"""
dtoz(Λ::DMAconfig, d) = dtoz(Λ, 1, d)

dtoz(Λ::DMAconfig, k, d) = k .* ec .* cc(Λ, d) ./ (3.0π .* η(Λ, Λ.gas) .* d)

@doc raw"""
    vtoz(Λ::DMAconfig, v)

Converts between voltage and selected mobility. It is the inverse of [ztov](@ref). 

For the cylindrical DMA and balanced flows: 

``z^s = \frac{q_{sh}}{2\pi l v} \ln \left(\frac{r_2}{r_1}\right)``

For the radial DMA and balanced flows:

``z^s = \frac{q_{sh} l}{\pi v \left({r_2}^2 - {r_1}^2\right)} ``

where ``v`` is the potential applied between the inner and out section of the annulus, 
``r_1``, ``r_2``, and ``l`` are the dimensions of the cylindrical DMA  and ``q_{sh}`` is 
the sheath flow rate.

Example Usage
```julia
t,p = 295.15, 1e5                        
qsa,qsh = 1.66e-5, 8.3e-5
r₁,r₂,l = 9.37e-3,1.961e-2,0.44369               
Λ = DMAconfig(t,p,qsa,qsh,r₁,r₂,l,0.0,:-,6,:cylindrical) 
mobility = vtoz(Λ,1000.0) # [m2 V-1 s-1]
```
"""
vtoz(Λ::DMAconfig, v) =
    (Λ.DMAtype == :radial) ? Λ.qsh .* Λ.l / (π .* (Λ.r2^2.0 - Λ.r1^2) .* v) :
    Λ.qsh ./ (2.0π .* Λ.l .* v) .* log(Λ.r2 / Λ.r1)

@doc raw"""
    ztov(Λ::DMAconfig, v)
    
Converts between selected mobility and voltage. It is the inverse of [vtoz](@ref). 

Example Usage
```julia
t,p = 295.15, 1e5                        
qsa,qsh = 1.66e-5, 8.3e-5
r₁,r₂,l = 9.37e-3,1.961e-2,0.44369               
Λ = DMAconfig(t,p,qsa,qsh,r₁,r₂,l,0.0,:-,6,:cylindrical) 
voltage = ztov(Λ,1e-9) 
```
"""
ztov(Λ::DMAconfig, z) =
    (Λ.DMAtype == :radial) ? Λ.qsh .* Λ.l / (π .* (Λ.r2^2.0 - Λ.r1^2) * z) :
    Λ.qsh ./ (2.0π .* Λ.l .* z) .* log(Λ.r2 / Λ.r1)

f(Λ, i, z, di) = @. i .* ec .* cc($Ref(Λ), di) ./ (3.0π .* η($Ref(Λ), $Ref(Λ.gas)) .* z)
converge(f, g) = maximum(abs.(1.0 .- f ./ g) .^ 2.0) < 1e-24
g(Λ, i, z, di) = converge(f(Λ, i, z, di), di) ? di : g(Λ, i, z, f(Λ, i, z, di))

@doc raw"""
    ztod(Λ::DMAconfig, i::Int, z)

Converts mobility to diameter.
- Λ is the DMA configuration
- i is the number of charges
- z is the mobility

```julia
t,p = 295.15, 1e5                        
qsa,qsh = 1.66e-5, 8.3e-5                     
r₁,r₂,l = 9.37e-3,1.961e-2,0.44369               
Λ = DMAconfig(t,p,qsa,qsh,r₁,r₂,l,0.0,:-,6,:cylindrical) 
z = dtoz(Λ,100.0*1e-9) 
diameter = ztod(Λ,1,z) 
```
"""
@memoize ztod(Λ::DMAconfig, i::Int, z) = g(Λ, i, z, di) .* 1e9;

@doc raw"""
    Ω(Λ::DMAconfig, Z, zs)

The DMA transfer function is the probability that a particle of a particle of a given size 
exits the classifier via the sample flow. The diffusive broadened DMA transfer function is 
computed assuming blanced sheath and excess flows using the expression of Stolzenburg 
and McMurry (2008).

``\Omega(\tilde{z},\beta,\sigma) = \frac{\sigma}{\sqrt{2}\beta}\left[\epsilon \left( 
    \frac{\tilde{z}-(1+\beta)}{\sqrt{2}\sigma} \right) + \epsilon \left (\frac{\tilde{z}-
    (1-\beta)}{\sqrt{2}\sigma} \right) - 2\epsilon \left 
    ( \frac{\tilde{z}-1}{\sqrt{2}\sigma}\right)  \right]``
        
where ``\tilde{z} = \frac{z}{z^s}`` is the dimensionless mobility, ``z`` is the particle 
mobility ``z^s`` is the centroid mobility selected by the DMA, 
``\epsilon = x \mathrm{erf}(x) +\left(\exp(-x^2)/\sqrt{\pi}\right)``, ``\mathrm{erf}`` is 
the error function, and ``\beta = \frac{q_{sa}}{q_{sh}}``. The parameter ``\sigma`` 
accounts for diffusional broading of the transfer function. Assuming plug flow, 
``\sigma`` can be computed using the following equations Hagwood (1999) 
    
``\gamma = \left(\frac{r_1}{r_2}\right)^2``

``I = \frac{1}{2}(1+γ)``

``\kappa = \frac{lr_2}{r_2^2-r_1^2}``
    
``G = \frac{4(1+\beta)^2}{(1-γ)} \left[I+\{2(1+\beta)\kappa\}^{-2} \right ]``
    
``\sigma = \sqrt{\frac{2G\pi ld_{ab}}{q_{sh}}}``
   
Inputs for flow are taken from the DMAconfig. The function expects a mobility scalar z or vector Z,
and a centroid mobility zˢ.

Example Usage
```julia
zˢ = dtoz(Λ, 200e-9)      # centroid mobility for Dp = 200 nm
z = [1e-9, 1e-8, 1e-7]    # mobility 
Ω(Λ,z,zˢ)                 # Output of the transfer function
```

!!! note

    The function Ω is embedded in the the Type DifferentialMobilityAnalyzers.jl, which 
    assigns δ.Ω either to this function Ω or Ωav applicable to scanning mode,

"""
function Ω(Λ::DMAconfig, Z, zs)
    ε = (x) -> @. x * erf(x) .+ exp(-x^2.0) / √π
    D = dab(Λ, ztod(Λ, 1, zs) * 1e-9)
    β = Λ.qsa / Λ.qsh
    γ = (Λ.r1 / Λ.r2)^2.0
    I = 0.5(1.0 + γ)
    κ = Λ.l * Λ.r2 / (Λ.r2^2.0 - Λ.r1^2.0)
    G = 4.0(1.0 + β)^2.0 / (1.0 - γ) * (I + (2.0(1.0 + β)κ)^(-2.0))
    σ = √(G * 2.0π * Λ.l * D / Λ.qsh)
    f =
        (Z, σ, β, ε) -> @. σ / (√2.0 * β) * (
            ε((Z - (1.0 + β)) / (√2.0 * σ)) + ε((Z - (1.0 - β)) / (√2.0 * σ)) -
            2.0 * ε((Z - 1.0) / (√2.0 * σ))
        )
    return clean(f(Z / zs, σ, β, ε))
end

function Ω(Λ::DMAconfig, Z, zs, k)
    ε = (x) -> @. x * erf(x) .+ exp(-x^2.0) / √π
    D = dab(Λ, ztod(Λ, k, zs) * 1e-9)
    β = Λ.qsa / Λ.qsh
    γ = (Λ.r1 / Λ.r2)^2.0
    I = 0.5(1.0 + γ)
    κ = Λ.l * Λ.r2 / (Λ.r2^2.0 - Λ.r1^2.0)
    G = 4.0(1.0 + β)^2.0 / (1.0 - γ) * (I + (2.0(1.0 + β)κ)^(-2.0))
    σ = √(G * 2.0π * Λ.l * D / Λ.qsh)
    f =
        (Z, σ, β, ε) -> @. σ / (√2.0 * β) * (
            ε((Z - (1.0 + β)) / (√2.0 * σ)) + ε((Z - (1.0 - β)) / (√2.0 * σ)) -
            2.0 * ε((Z - 1.0) / (√2.0 * σ))
        )
    return clean(f(Z / zs, σ, β, ε))
end

function mylogspace(a::Float64, b::Float64, n::Int)
    x = Float64[]
    push!(x, a)
    step = 10.0^(log10(b / a) / (n - 1))
    for i = 2:1:n
        push!(x, x[i-1] * step)
    end
    return x
end


@doc raw"""
    setupSMPS(Λ::DMAconfig, v1::Number, v2::Number, tscan::Number, tc::Number)

Construct the [DifferentialMobilityAnalyzer](@ref) type for DMA configuration
[DMAconfig](@ref). The size grid is constructed between voltage v1 and v2, tscan is 
the duration of the SMPS scan in seconds, tc is the integration time per bin. The
number of bins is given by tscan / tc. Per convenction instantiations of this type are
denoted as δ, or δ₁, δ₂ ... to distinguish DMA chains. The grid must be setup from low 
voltage to high voltage. The grid is then setup in order from high to low diameter.

Diameters stored in δ are in units of nm.

```julia
t,p = 295.15, 1e5                        
qsa,qsh = 1.66e-5, 8.3e-5                     
r₁,r₂,l = 9.37e-3,1.961e-2,0.44369               
Λ = DMAconfig(t,p,qsa,qsh,r₁,r₂,l,0.0,:-,6,:cylindrical) 

δ = setupSMPS(Λ, 10, 10000, 180, 1.5)
```
"""
@memoize function setupSMPS(Λ::DMAconfig, v1::Number, v2::Number, tscan::Number, tc::Number)
    bins = round(Int, tscan / tc) # Number of size bins
    Ve = reverse(10 .^ range(log10(v1), stop = log10(v2), length = bins + 1))  # Voltage bin-edges
    Vp = sqrt.(Ve[2:end] .* Ve[1:end-1])  # Voltage midpoints
    Tc = getTc(Λ)
    Ze = vtoz(Λ, Ve)
    Z = vtoz(Λ, Vp)
    Dp = ztod(Λ, 1, Z)
    De = ztod(Λ, 1, Ze)
    ΔlnD = log.(De[1:end-1] ./ De[2:end])

    function Ωav(Λ::DMAconfig, i::Int, k::Int; nint = 20)
        Vex = mylogspace(Ve[i], Ve[i+1], nint)
        return mapreduce(zˢ -> Ω(Λ, Z, zˢ / k, k), +, vtoz(Λ, Vex)) / nint
    end

    T = (i, k, Λ) -> Ωav(Λ, i, k) .* Tc(k, Dp) .* Tl(Λ, Z, k)
    𝐀 = (hcat(map(i -> Σ(k -> T(i, k, Λ), Λ.m), 1:bins)...))'
    𝐎 = (hcat(map(i -> Σ(k -> Ωav(Λ, i, k) .* Tl(Λ, Dp), 1), 1:bins)...))'
    𝐈 = Matrix{Float64}(I, bins, bins)
    n, m = size(𝐀)
    𝐒 = zeros(n, m)
    for i = 1:n
        𝐒[i, i] = sum(𝐀[i, :])
    end

    return DifferentialMobilityAnalyzer(Ωav, Tc, Tl, Z, Ze, Dp, De, ΔlnD, 𝐀, 𝐒, 𝐎, 𝐈)
end

@doc raw"""
    setupSMPSdata(Λ::DMAconfig, V::AbstractVector)

Construct the [DifferentialMobilityAnalyzer](@ref) type for DMA configuration
[DMAconfig](@ref). The size grid is constructed for a vector of voltages sorted from
low to high. The voltage correspond to bin edges and might correspond to gridded data
output obtained from an SMPS. The number of bins is bins = length(V)-1. 

Diameters stored in δ are in units of nm.

```julia
t,p = 295.15, 1e5                        
qsa,qsh = 1.66e-5, 8.3e-5                     
r₁,r₂,l = 9.37e-3,1.961e-2,0.44369               
Λ = DMAconfig(t,p,qsa,qsh,r₁,r₂,l,0.0,:-,6,:cylindrical) 

V = range(10, stop = 10000, length=121)
δ = setupSMPSdata(Λ, V)
```
"""
@memoize function setupSMPSdata(Λ::DMAconfig, V::AbstractVector)
    tc = 1
    Ve = (V[1] < V[2]) ? reverse(V) : V
    bins = length(Ve) - 1
    Vp = sqrt.(Ve[2:end] .* Ve[1:end-1])  # Voltage midpoints
    Tc = getTc(Λ)
    Ze = vtoz(Λ, Ve)
    Z = vtoz(Λ, Vp)
    Dp = ztod(Λ, 1, Z)
    De = ztod(Λ, 1, Ze)
    ΔlnD = log.(De[1:end-1] ./ De[2:end])

    function Ωav(Λ::DMAconfig, i::Int, k::Int; nint = 20)
        Vex = mylogspace(Ve[i], Ve[i+1], nint)
        return mapreduce(zˢ -> Ω(Λ, Z, zˢ / k, k), +, vtoz(Λ, Vex)) / nint
    end

    T = (i, k, Λ) -> Ωav(Λ, i, k) .* Tc(k, Dp) .* Tl(Λ, Z, k)
    𝐀 = (hcat(map(i -> Σ(k -> T(i, k, Λ), Λ.m), 1:bins)...))'
    𝐎 = (hcat(map(i -> Σ(k -> Ωav(Λ, i, k) .* Tl(Λ, Dp), 1), 1:bins)...))'
    𝐈 = Matrix{Float64}(I, bins, bins)
    n, m = size(𝐀)
    𝐒 = zeros(n, m)
    for i = 1:n
        𝐒[i, i] = sum(𝐀[i, :])
    end
    𝐒⁺ = inv(𝐒)
    return DifferentialMobilityAnalyzer(Ωav, Tc, Tl, Z, Ze, Dp, De, ΔlnD, 𝐀, 𝐒, 𝐎, 𝐈)
end

@doc raw"""
    setupDMA(Λ::DMAconfig, z1::Number, z2::Number, bins::Int)

Construct the [DifferentialMobilityAnalyzer](@ref) type for DMA configuration
[DMAconfig](@ref). The size grid is constructed between mobility z1 and z2, with 
bin + 1 edges and bin number of midpoints. Per convenction instantiations of this
type are denoted as δ, or δ₁, δ₂ ... to distinguish DMA chains. The grid must be 
setup from low to high mobility, corresponding to large to small mobility diameter.

Diameters stored in δ are in units of nm.

Example Usage
```julia
t,p = 295.15, 1e5                        
qsa,qsh = 1.66e-5, 8.3e-5                     
r₁,r₂,l = 9.37e-3,1.961e-2,0.44369               
Λ = DMAconfig(t,p,qsa,qsh,r₁,r₂,l,0.0,:-,6,:cylindrical) 
bins,z₁,z₂ = 60, vtoz(Λ,10000), vtoz(Λ,10)   

δ = setupDMA(Λ, z₁, z₂, bins)
```
"""
@memoize function setupDMA(Λ::DMAconfig, z1::Number, z2::Number, bins::Int)
    Tc = getTc(Λ)
    Ze = 10 .^ range(log10(z1), stop = log10(z2), length = bins + 1)
    Z = sqrt.(Ze[2:end] .* Ze[1:end-1])
    Dp = ztod(Λ, 1, Z)
    De = ztod(Λ, 1, Ze)
    ΔlnD = log.(De[1:end-1] ./ De[2:end])
    T = (zˢ, k, Λ) -> Ω(Λ, Z, zˢ / k, k) .* Tc(k, Dp) .* Tl(Λ, Z, k)
    𝐀 = (hcat(map(zˢ -> Σ(k -> T(zˢ, k, Λ), Λ.m), Z)...))'
    𝐎 = (hcat(map(i -> Σ(k -> Ω(Λ, Z, i / k) .* Tl(Λ, Dp), 1), Z)...))'
    𝐈 = Matrix{Float64}(I, bins, bins)
    n, m = size(𝐀)
    𝐒 = zeros(n, m)
    for i = 1:n
        𝐒[i, i] = sum(𝐀[i, :])
    end
    𝐒⁺ = inv(𝐒)
    return DifferentialMobilityAnalyzer(Ω, Tc, Tl, Z, Ze, Dp, De, ΔlnD, 𝐀, 𝐒, 𝐎, 𝐈)
end

@memoize function setupDMAgridded(Λ, De)
    Tc = getTc(Λ)
    Ze = map(D -> dtoz(Λ, D * 1e-9), De)
    Z = sqrt.(Ze[2:end] .* Ze[1:end-1])
    Dp = ztod(Λ, 1, Z)
    ΔlnD = log.(De[1:end-1] ./ De[2:end])
    T = (zˢ, k, Λ) -> Ω(Λ, Z, zˢ / k, k) .* Tc(k, Dp) .* Tl(Λ, Z, k)
    𝐀 = (hcat(map(zˢ -> Σ(k -> T(zˢ, k, Λ), Λ.m), Z)...))'
    𝐎 = (hcat(map(i -> Σ(k -> Ω(Λ, Z, i / k) .* Tl(Λ, Dp), 1), Z)...))'
    𝐈 = Matrix{Float64}(I, length(Z), length(Z))
    n, m = size(𝐀)
    𝐒 = zeros(n, m)
    for i = 1:n
        𝐒[i, i] = sum(𝐀[i, :])
    end
    𝐒⁺ = inv(𝐒)
    return DifferentialMobilityAnalyzer(Ω, Tc, Tl, Z, Ze, Dp, De, ΔlnD, 𝐀, 𝐒, 𝐎, 𝐈)
end
