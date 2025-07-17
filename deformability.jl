# This script analyses the deformation of GUVs under magnetic fields
# ===


# Importing packages
# ---
# ! Please run _init_.jl to initialize the computing environment !
using CSV
using DataFrames
using Plots
using Distributions
using HypothesisTests
using StatsPlots
import PowerAnalyses as PA

# Utilities
pfs(x; s=1) = round(x,sigdigits=s);
pfd(x; d=1) = round(x,digits=d);

# Extended Data folder
# ---
ed_folder = "Extended_Data";
# Create the folder if it does not exist
if !isdir(ed_folder)
    mkpath(ed_folder)
end;
def_size_folder = "deformation_size";
# Create the folder if it does not exist
if !isdir(joinpath(ed_folder,def_size_folder))
    mkpath(joinpath(ed_folder,def_size_folder))
end;
sd_mag_folder = "size_distribution_magnetic";
# Create the folder if it does not exist
if !isdir(joinpath(ed_folder,sd_mag_folder))
    mkpath(joinpath(ed_folder,sd_mag_folder))
end;

# Importing data and calculating deformations
# ===

# Import data
# ---
dataset_names = ("GUVs_without_cholesterol_1","GUVs_without_cholesterol_2","GUVs_without_cholesterol_3","GUVs_with_cholesterol_6040_1","GUVs_with_cholesterol_6040_2","GUVs_with_cholesterol_6040_3");
datafile_names = dataset_names.*".csv";
data = CSV.read.(joinpath.("data",datafile_names), DataFrame, ntasks = 1, header=1);
# Magnetic fields
H = [2.0, 4.7, 48.0]; # mT

# Geometrical calculation functions
# ---
# a and b are the major and minor hemiaxes - assuming GUVs to be prolate spheroid under magnetic field
# Volume
V(a,b) = 4π*(a*b^2)/3;
# Equivalent radius and diameter
R(a,b) = (a*b^2)^(1/3);
D(a,b) = 2*R(a,b);
# Surface area at rest (sphere)
Ss(a,b) = 4π*R(a,b)^2;
# Surface area under magnetic field (ellipsoid / prolate spheroid)
ecc(a,b) = sqrt(1-b^2/a^2);
Sps(a,b) = 2π*b^2 * (1+(a/(b*ecc(a,b)))*asin(ecc(a,b))); 
# Surface area deformation
sigma(a,b) = Sps(a,b)/Ss(a,b) - 1;

# Add fields and geometrical features to dataframes
# ---
geom_functions = (V,R,D,Ss,Sps,sigma);
units = ("um3","um","um","um2","um2","");
function add_fields_geom!(df,field)
    df.field .= "H$field"
    for (f,u) in zip(geom_functions,units)
        df[!,string(f)*"_"*u] = f.(df.a_um,df.b_um)
    end
end;
add_fields_geom!.(data, repeat(1:3,2));
df_1000 = vcat(data[1:3]...,cols=:union);
df_6040 = vcat(data[4:6]...,cols=:union);

# Show and save dataframes
# ---
println("GUVs without cholesterol")
display(df_1000)
println("GUVs with cholesterol (DOPC:Chol 60:40)")
display(df_6040)
# Save dataframes
CSV.write(joinpath("results","GUVs_deformability_100_0.csv"),df_1000);
CSV.write(joinpath("results","GUVs_deformability_60_40.csv"),df_6040);

# Create grouped dataframes by magnetic field
# ---
gdf_1000 = groupby(df_1000,:field);
gdf_6040 = groupby(df_6040,:field);


# Size distributions of magnetic GUVs with and without cholesterol
# ===

# Histograms and distributions
# ---
hist_lim = max(ceil(maximum(maximum,[df_1000[!,:D_um],df_6040[!,:D_um]]),sigdigits=2),100) |> Int;
sample_n = ("100:0","60:40");
colors=(1,3);
fa=(0.4,0.8);
# Functions for plotting histograms and fitting LogNormal distribution
fitLN(x) = fit(LogNormal,x);
function HistLNDist(diameters,sample_n;color,fa)
    d_step = 5;
    b_range = 0:d_step:hist_lim
    N = length(diameters)
    h = histogram(diameters,bins=b_range, label="data (N=$N)", color=color, fillalpha=fa,linewidth=0)
    xlabel!("diameters (μm)")
    ylabel!("counts")
    ylims!((0,125))
    d = fitLN(diameters)
    plot!(x->pdf(d,x)*N*d_step,0,hist_lim,label="LogNormal",color=color,linewidth=2)
    peak_d = mode(d)
    scatter!([peak_d], [pdf(d,peak_d)]*N*d_step,label="$(round(peak_d,digits=1)) (μm)", color=color)
    title!(sample_n)
    return d, h, N
end;
function plotHists(diameters_1,diameters_2,sample_n)
    dists_1, h_1, N_1 = HistLNDist(diameters_1,sample_n[1],color=colors[1],fa=0.5)
    dists_2, h_2, N_2 = HistLNDist(diameters_2,sample_n[2],color=colors[2],fa=0.5)
    p = plot(h_1,h_2,layout=(2,1))
    display(p)
    return (dists_1,dists_2,N_1,N_2)
end;
# Plot histogram and fitting LogNormal distribution for each sample (at all fields)
dists_Ns = plotHists(df_1000.D_um,df_6040.D_um,sample_n)
# Save histograms
savefig(joinpath(ed_folder,sd_mag_folder,"sd_mag_all.svg"));

# Statistical significance of difference in size distribution (rest diameters)
# ---
# Type-I error rate
α=0.005;
# Format p value
format_p(p) = p < eps(Float64) ? 0.0 : pfs(p; s=2);
# Function for testing the statistical significance of mean differences between samples (assuming LogNormal distribution)
function StatSigDiffLN(xLN1,xLN2)
    xN1, xN2 = log.(xLN1), log.(xLN2)
    Ftest = VarianceFTest(xN1,xN2)
    if pvalue(Ftest) < 0.05
        ttest = UnequalVarianceTTest(xN1,xN2)
    else
        ttest = EqualVarianceTTest(xN1,xN2)
    end
    return ttest
end;
# Test the statistical significance of size distribution differences between samples (assuming LogNormal distribution)
# All field values
sigdiff_tot_ttest = StatSigDiffLN(df_1000.D_um,df_6040.D_um);
println("Statistically significant difference in size distribution, all fields (α = $α): $(pvalue(sigdiff_tot_ttest)<α)")
println("p-value: $(format_p(pvalue(sigdiff_tot_ttest)))")
# Individual field values
sigdiff_h_ttests = Vector{Any}(undef,3);
for i in 1:3
    plotHists(gdf_1000[i].D_um,gdf_6040[i].D_um,sample_n.*", H$i")
    savefig(joinpath(ed_folder,sd_mag_folder,"sd_mag_H$i.svg"))
    sigdiff_h_ttests[i] = StatSigDiffLN(gdf_1000[i].D_um,gdf_6040[i].D_um)
    println("Statistically significant difference in size distribution, H$i (α = $α): $(pvalue(sigdiff_h_ttests[i])<α)")
    println("p-value: $(format_p(pvalue(sigdiff_h_ttests[i])))")
end;
df_stat = DataFrame(field = ["H1","H2","H3"])
df_stat[!, "p value size"] = [format_p(pvalue(sigdiff_h_ttests[i])) for i in 1:3];


# Deformation of magnetic GUVs with and without cholesterol
# ===

# σ vs. equivalent radius
for i in 1:3
    p = @df gdf_1000[i] scatter(:R_um,:sigma_,markercolor=colors[1], markerstrokewidth=0, label="100:0, H$i")
    
    @df gdf_6040[i] scatter!(:R_um,:sigma_,markercolor=colors[2], markerstrokewidth=0, label="60:40, H$i")

    xlabel!("R (μm)")
    ylabel!("σ = Sₚₛ/Sₛ − 1")
    # ylabel!("σ⋅R = (Sₚₛ/Sₛ − 1)⋅R")
    xlims!(0,35)
    ylims!(-0.01,0.45)
    # ylims!(-Inf,3)
    display(p)
    savefig(joinpath(ed_folder,def_size_folder,"sd_deform_scatter_H$i.svg"))
end

# Testing statistical significance
# ---
# Test the statistical significance of differences between samples (assuming LogNormal distribution of σ) for each magnetic field
sigdiff_S_ttests = Vector{Any}(undef,3);
for i in 1:3
    sigdiff_S_ttests[i] = StatSigDiffLN(gdf_1000[i].sigma_,gdf_6040[i].sigma_)
    println("Statistically significant difference in surface area deformation σ, H$i (α = $α): $(pvalue(sigdiff_S_ttests[i])<α)")
end
df_stat[!, "p value σ (t test - LN)"] = @. format_p(pvalue(sigdiff_S_ttests))
function effect_size_LN(xLN1,xLN2)
    x1, x2 = log.(xLN1), log.(xLN2)
    return (mean(x1) - mean(x2)) / sqrt((var(x1) + var(x2)) / 2)
end
sigma_factor(es,xLN1,xLN2) = exp(es*sqrt((var(log.(xLN1)) + var(log.(xLN2))) / 2))
# Calculate the minimum effect size detectable with given power for all fields
power = 0.8
es_all = PA.get_es(PA.IndependentSamplesTTest(PA.Tail(1)), power=power, alpha=α, n=min(length(df_1000.sigma_), length(df_6040.sigma_)))
sigma_factor_es_all = sigma_factor(es_all,df_1000.sigma_,df_6040.sigma_)
# Calculate the minimum effect size detectable with given power for each field
es_fields = [PA.get_es(PA.IndependentSamplesTTest(PA.Tail(1)), power=power, alpha=α, n=min(length(d1.sigma_), length(d2.sigma_))) for (d1,d2) in zip(gdf_1000,gdf_6040)]
sigma_factor_es_fields = [sigma_factor(es, d1.sigma_, d2.sigma_) for (d1,d2,es) in zip(gdf_1000,gdf_6040,es_fields)]
# Add minimum detectable effect size to the dataframe
df_stat[!, "min detectable σ1/σ2 (t test - LN - power=$power)"] = pfs.(sigma_factor_es_fields,s=3)
# Save statistical significance results
CSV.write(joinpath("results","GUVs_deformability_stat.csv"),df_stat);

# Test the statistical significance of differences between fields
combos = ((1,2),(1,3),(2,3));
println("Statistically significant difference in surface area deformation σ (α = $α):")
for c in combos
    ttest_1000 = StatSigDiffLN(gdf_1000[c[1]].sigma_,gdf_1000[c[2]].sigma_)
    ttest_6040 = StatSigDiffLN(gdf_6040[c[1]].sigma_,gdf_6040[c[2]].sigma_)
    println("H$c: 100:0 → $(pvalue(ttest_1000)<α), 60:40 → $(pvalue(ttest_6040)<α)")
end

# Generating Figure 5
# ---
begin
    def = @df df_1000 violin(:field,:sigma_, side=:left,label=sample_n[1], linewidth=0, fillalpha=fa[1]/2,fillcolor=colors[1], size=(400,300))
    @df df_6040 violin!(:field,:sigma_, side=:right,label=sample_n[2], linewidth=0, fillalpha=fa[1]/2,fillcolor=colors[2])
    @df df_1000 dotplot!(:field,:sigma_, side=:left, markercolor=colors[1], markerstrokewidth=0, markersize=2, markeralpha=fa[2], label="")
    @df df_6040 dotplot!(:field,:sigma_, side=:right, markercolor=colors[2], markerstrokewidth=0, markersize=2, markeralpha=fa[2], label="")
    ylims!(-0.01,0.45)
    ylabel!("σ = Sₚₛ/Sₛ − 1")
    display(def)
end
savefig(joinpath("results","Figure5.svg"))