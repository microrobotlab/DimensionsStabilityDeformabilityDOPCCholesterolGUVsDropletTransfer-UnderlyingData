# This script analyses the size distribution of GUVs at t₀ and after overnight storage
# ===


# Importing packages
# ---
# ! Please run _init_.jl to initialize the computing environment !
using Statistics
using CSV
using DataFrames
using Plots
using Distributions
using HypothesisTests
using Combinatorics
using StatsPlots
using Measures


# Extended Data folder
# ---
ed_folder = "Extended_Data";
# Create the folder if it does not exist
if !isdir(ed_folder)
    mkpath(ed_folder)
end;
sd_stab_folder = "size_distribution_T0vsON";
# Create the folder if it does not exist
if !isdir(joinpath(ed_folder,sd_stab_folder))
    mkpath(joinpath(ed_folder,sd_stab_folder))
end;

# Importing, showing, and fitting size distribution data
# ===

# Importing data
# ---
# Data obtained by microscopy images acquired at t₀ and after o.n. storage and manual measuring with ImageJ (measures in pixels)
t₀_dataset_name = "GUVs_stability_T0";
on_dataset_name = "GUVs_stability_ON";
t₀_datafile_name, on_datafile_name = [t₀_dataset_name,on_dataset_name].*".csv";
t₀_data, on_data = CSV.read.(joinpath.("data",[t₀_datafile_name,on_datafile_name]), DataFrame, ntasks = 1, header=1);


# Calculating diameters (μm) and surface area (μm²) from area (pixels) - assuming spherical GUVs
# ---
# Scaling factor
conv_pix_μm = 8.2978;   # pix/μm

# Function for extracting and scaling diameter data
diameter_μm(area_px) = 2*sqrt(area_px/π)/conv_pix_μm;
# Function to add diameters to dataframe
function add_diameters_μm!(dataframe)
    dataframe[!,:Diameter_um] = diameter_μm.(dataframe[!,:Area_px])
end;
# Add diameters to dataframe (μm)
add_diameters_μm!.([t₀_data,on_data]);

# Function to calculate surface area from diameter
surfarea(diameter) = π*diameter^2;
# Function to add surface areas to dataframe
function add_surfarea!(df)
    df[!,:SurfArea_um2] = surfarea.(df[!,:Diameter_um])
end;
# Add surface areas to dataframe (μm²)
add_surfarea!.([t₀_data,on_data]);


# Sorting and grouping data by DOPC:Chol ratio
# ---
# Get sample names / concentrations from Label
sample_n = unique(t₀_data[!,:Label]);
conc_names = Dict("60_40"=>"60:40","71_29"=>"71:29","85_15"=>"85:15","NC"=>"100:0");
# Sort by concentration Label
sort!(t₀_data,:Label,rev=true);
sort!(on_data,:Label,rev=true);
# Create grouped dataframes from sample name (DOPC:Chol ratio)
gdf_t₀, gdf_on = groupby.([t₀_data,on_data],:Label, sort=false);
gdfs = zip(gdf_t₀, gdf_on);


# Showing data tables
# ---
println("t₀ data")
display(t₀_data)
println("o.n. data")
display(on_data)

# Save dataframes to CSV files
# ---
CSV.write(joinpath("results",t₀_dataset_name*"_proc.csv"), t₀_data);
CSV.write(joinpath("results",on_dataset_name*"_proc.csv"), on_data);


# Plotting size distributions data
# ---
# Utilities
hist_lim = max(ceil(maximum(maximum,[t₀_data.Diameter_um...,on_data.Diameter_um...]),sigdigits=2),100) |> Int;
colors_on = cgrad([palette(:default)[3]/1.5, palette(:default)[1]]/1.5,[0.6,0.71,0.85,1.0],categorical=true)[4:-1:1];
colors_t₀ = cgrad([palette(:default)[3], palette(:default)[1]],[0.6,0.71,0.85,1.0],categorical=true)[4:-1:1];
pfs(x; s=1) = round(x,sigdigits=s);
pfd(x; d=1) = round(x,digits=d);
# Function for creating the histogram and fitting LogNormal distribution to each sample's data
function HistLNDist(diameters;color,fa)
    d_step = 5
    b_range = 0:d_step:hist_lim
    N = length(diameters)
    h = histogram(diameters,bins=b_range, label="data (N=$N)", color=color, fillalpha=fa,linewidth=0) # normalize=:pdf, 
    xlabel!("diameters (μm)")
    ylabel!("counts")
    ylims!((0,165))#0.1
    xlims!((0,hist_lim))
    d = fit(LogNormal,diameters)
    plot!(x->pdf(d,x)*N*d_step,0,hist_lim,label="LogNormal",color=color,linewidth=2)
    peak_d = mode(d)
    scatter!([peak_d], [pdf(d,peak_d)*N*d_step],label="$(round(peak_d,digits=1)) (μm)", color=color)
    return d, h, N
end;
# Function for plotting the histograms and returning the distribution and number of GUVs
function plotHists(gs)
    function plotHist(g,c,t)
        dist, h, N = HistLNDist(g.Diameter_um,color=c,fa=0.5)
        title!(h,"$(conc_names[g.Label[1]]), $t")
        return h, dist, N
    end
    cidx = findfirst(sample_n.==gs[1].Label[1])
    t₀, on = plotHist.(gs,[colors_t₀[cidx],colors_on[cidx]],["t₀","o.n."])
    p = plot(t₀[1],on[1],layout=(2,1))
    savefig(p, joinpath(ed_folder,sd_stab_folder,"sd_size_dist_"*gs[1].Label[1]*".svg"))
    display(p)
    return(t₀[2:3]...,on[2:3]...,gs[1].Label[1])
end;
# Plot histogram and fit LogNormal distribution for each sample (Extended Data figures)
dists_Ns = plotHists.(gdfs);
# Showing medians from fitted LogNormal distributions
df_conc = DataFrame();
df_conc[!,"DOPC:Chol"] = [conc_names[d[5]] for d in dists_Ns];
df_conc[!,"t₀ median (μm)"] = [median(d[1]) |> pfd for d in dists_Ns];
df_conc[!,"t₀ peak (μm)"] = [mode(d[1]) |> pfd for d in dists_Ns];
df_conc[!,"t₀ N"] = [d[2] for d in dists_Ns];
df_conc[!,"o.n. median (μm)"] = [median(d[3]) |> pfd for d in dists_Ns];
df_conc[!,"o.n. peak (μm)"] = [mode(d[3]) |> pfd for d in dists_Ns];
df_conc[!,"o.n. N"] = [d[4] for d in dists_Ns];
# CSV.write(joinpath("results","GUVs_stability_size_distribution.csv"),df_conc);
# The median of a LogNormal distribution coincides with exp(mean(log(diameters)))


# Statistical analysis
# ===
# Core function for testing the statistical significance of mean differences assuming LogNormal distributions
function StatSigDiffLN(xLN1,xLN2)
    xN1, xN2 = log.(xLN1), log.(xLN2)
    Ftest = VarianceFTest(xN1,xN2)
    if pvalue(Ftest) < 0.05
        ttest = UnequalVarianceTTest(xN1,xN2)
    else
        ttest = EqualVarianceTTest(xN1,xN2)
    end
    return Ftest, ttest
end;

# Statistical significance of size distribution differences
# ---
# Function for testing statistical significance of mean differences between samples (assuming LogNormal distributions) for each combination of samples
function test_SSD_LN(gdf,XLN,combinations)
    Ftests = []
    ttests = []
    for c in combinations
        Ftest, ttest = StatSigDiffLN(gdf[c[1]][!,Symbol(XLN)],gdf[c[2]][!,Symbol(XLN)])
        push!(Ftests,Ftest)
        push!(ttests,ttest)
    end
    # unequal_variances = findall(pvalue.(Ftests).<0.05)
    # sig_mean_diff = findall(pvalue.(ttests).<0.05)
    return ttests #sig_mean_diff
end;
# Function for testing  the statistical significance of mean differences between pristine and overnight samples (assuming LogNormal distributions)
function test_cross_SSD_LN(gdf1,gdf2,XLN)
    Ftests = []
    ttests = []
    for (g1,g2) in zip(gdf1, gdf2)
        Ftest, ttest = StatSigDiffLN(g1[!,Symbol(XLN)],g2[!,Symbol(XLN)])
        push!(Ftests,Ftest)
        push!(ttests,ttest)
    end
    # unequal_variances = findall(pvalue.(Ftests).<0.05)
    # sig_mean_diff = findall(pvalue.(ttests).<0.05)
    return ttests #sig_mean_diff
end;
# Type-I error rate
α=0.005;
# Format p value
format_p(p) = p < eps(Float64) ? 0.0 : pfs(p; s=2);

# Effects of DOPC:Chol ratio on GUVs stability
# ---
# Test the statistical significance of differences between t₀ and o.n. samples
ttests_stability = test_cross_SSD_LN(gdf_t₀,gdf_on,:Diameter_um);
p_stability = pvalue.(ttests_stability);
df_conc[!,"p value (t₀ vs o.n.)"] = format_p.(p_stability);
# df_conc[!,"significance - α = $α"] = p_stability.<α;
# println("Significantly different samples t₀ vs. o.n.: $(sample_n[sig_mean_diff_cross])")

# Calculate total surface area of GUVs
tot_surf_area = combine.([gdf_t₀,gdf_on],:SurfArea_um2 => sum) |> x -> innerjoin(x...,on=:Label,renamecols="_t0"=>"_on");
# Calculate ratio of surface area between o.n. samples and t₀ samples
tot_surf_area.ratio_on_t₀ = tot_surf_area.SurfArea_um2_sum_on ./ tot_surf_area.SurfArea_um2_sum_t0;
# Add surface area ratios to the dataframe
df_conc[!,"Surface ratio o.n./t₀ (%)"] = round.(Int,100tot_surf_area.ratio_on_t₀);
# Save the dataframe with p values, statistical significance, and surface area ratios
CSV.write(joinpath("results","GUVs_stability_T0vsON.csv"),df_conc);   
# println("Surface ratio o.n./t₀: ")
# for r in eachrow(tot_surf_area) 
#     println("$(conc_names[r.Label]) => $(r.ratio_on_t₀)")
# end

# Effects of DOPC:Chol ratio on GUVs size distributions
# ---
# Get all combinations of samples (combinations are the same for t₀ and on)
combos = collect(combinations(eachindex(sample_n),2));
combos_sample_n = [sample_n[c[:]] for c in combos];
# Create the dataframe collecting the results
df_stat_conc = DataFrame();
df_stat_conc[!,"Sample A"]=[conc_names[c[1]] for c in combos_sample_n];
df_stat_conc[!,"Sample_B"]=[conc_names[c[2]] for c in combos_sample_n];
# Test the statistical significance of differences among t₀ samples
ttests_t₀ = test_SSD_LN(gdf_t₀,:Diameter_um,combos);
p_t0 = pvalue.(ttests_t₀);
df_stat_conc[!,"p value (t₀)"] = format_p.(p_t0);
# sig_mean_diff_t₀ = findall(pvalue.(ttests_t₀).<α);
# println("Significantly different combinations (α = $α) at t₀: $(combos_sample_n[sig_mean_diff_t₀])")
# df_stat_conc[!,"significance (t₀) - α = $α"] = p_t0.<α;
# Test the statistical significance of differences among o.n. samples
ttests_on = test_SSD_LN(gdf_on,:Diameter_um,combos);
p_on = pvalue.(ttests_on);
df_stat_conc[!,"p value (o.n.)"] = format_p.(p_on);
# sig_mean_diff_on = findall(pvalue.(ttests_on).<α);
# println("Significantly different combinations (α = $α)  o.n.: $(combos_sample_n[sig_mean_diff_on])")
# df_stat_conc[!,"significance (o.n.) - α = $α"] = p_on.<α;
CSV.write(joinpath("results","GUVs_concentration_stability.csv"),df_stat_conc);

# Generation of Figure 3
# ===

# Panel A
# ---
begin
    p1 = plot()
    favp = 0.3

    for gs in gdfs
        cidx = findfirst(sample_n.==gs[1].Label[1])
        violin!([gs[1].Label[1]], gs[1].Diameter_um, side=:left, linewidth=0, fillalpha=favp, color=colors_t₀[cidx],label="")
        violin!([gs[2].Label[1]], gs[2].Diameter_um, side=:right, linewidth=0, fillalpha=favp, color=colors_on[cidx],label="")
        dotplot!([gs[1].Label[1]], gs[1].Diameter_um, side=:left, markercolor=colors_t₀[cidx], markerstrokewidth=0, markersize=1, markeralpha=0.7, label="")
        dotplot!([gs[2].Label[1]], gs[2].Diameter_um, side=:right, markercolor=colors_on[cidx], markerstrokewidth=0, markersize=1, markeralpha=0.7, label="")
    end

    xticks!(xticks(p1)[1][1],[conc_names[xt] for xt in xticks(p1)[1][2]])
    xlabel!("DOPC:cholesterol in LS")
    ylabel!("measured diameters (μm)")
    ylims!(0,150)
    
    scatter!([],[],markershape=:ltriangle, markercolor=mean(colors_t₀),markerstrokewidth=0,alpha=0.7,label="t₀")  
    scatter!([],[],markershape=:rtriangle, markercolor=mean(colors_on),markerstrokewidth=0,alpha=0.7,label="o.n.")
end

# Panel B
# ---
begin
    p2 = bar(0.8:3.8,[d[2] for d in dists_Ns],bar_width=0.4, linewidth=0, fillalpha=0.7, color=colors_t₀, label="")
    bar!(1.2:4.2, [d[4] for d in dists_Ns],bar_width=0.4, linewidth=0, fillalpha=0.7, color=colors_on, label="")
    xticks!(1:4,[conc_names[d[5]] for d in dists_Ns])
    # xlabel!("DOPC:cholesterol in LS")
    ylabel!("# of vesicles")
    scatter!([],[],markershape=:ltriangle, markercolor=mean(colors_t₀),markerstrokewidth=0,alpha=0.7,label="t₀")  
    scatter!([],[],markershape=:rtriangle, markercolor=mean(colors_on),markerstrokewidth=0,alpha=0.7,label="o.n.")
end


# Panel C
# ---
# Get relevant quantiles from fitted distributions
yeLN(d) = (median(d) - quantile(d,0.05),quantile(d,0.95)-median(d))
yerr_d(d) = -exp(meanlogx(d)-stdlogx(d))+exp(meanlogx(d));
yerr_u(d) = exp(meanlogx(d)+stdlogx(d))-exp(meanlogx(d));
yerrors_t₀ = ([yerr_d(d[1]) for d in dists_Ns],[yerr_u(d[1]) for d in dists_Ns]);
yerrors_on = ([yerr_d(d[3]) for d in dists_Ns],[yerr_u(d[3]) for d in dists_Ns]);
# Plot
begin
    xl = 0.8:3.8;
    xr = 1.2:4.2;
    p3 = bar(xl,zeros(4),bar_width=0.4, linewidth=0, fillalpha=0.5, color=colors_t₀, label="")
    bar!(xr, zeros(4),bar_width=0.4, linewidth=0, fillalpha=0.7, color=colors_on, label="")
    for di in eachindex(dists_Ns)
        d_t₀, d_on = dists_Ns[di][1], dists_Ns[di][3]
        # Median & quantiles, t₀
        plot!([xl[di]-0.2,xl[di]+0.2],[median(d_t₀),median(d_t₀)],ribbon=([yeLN(d_t₀)[1]],[yeLN(d_t₀)[2]]),color=colors_t₀[di],lw=2,label="")
        # Median & quantiles, on
        plot!([xr[di]-0.2,xr[di]+0.2],[median(d_on),median(d_on)],ribbon=([yeLN(d_on)[1]],[yeLN(d_on)[2]]),color=colors_on[di],lw=2,label="")
        # Peak diameter (mode), t₀
        plot!([xl[di]-0.2,xl[di]+0.2],[mode(d_t₀),mode(d_t₀)],color=colors_t₀[di],ls=:dash,label="",alpha=1)
        # Peak diameter (mode), o.n.
        plot!([xr[di]-0.2,xr[di]+0.2],[mode(d_on),mode(d_on)],color=colors_on[di],ls=:dash,label="",alpha=1)
    end        
    xticks!(1:4,[conc_names[d[5]] for d in dists_Ns])
    xlabel!("DOPC:cholesterol in LS")
    ylabel!("diameters distribution (μm)")
    ylims!(0,80)
    scatter!([],[],markershape=:ltriangle, markercolor=mean(colors_t₀),markerstrokewidth=0,alpha=0.7,label="t₀",legend=:topleft)  
    scatter!([],[],markershape=:rtriangle, markercolor=mean(colors_on),markerstrokewidth=0,alpha=0.7,label="o.n.")
end

# Compose Figure
# ---
function labelplot(label)
    lp = plot(grid = false, showaxis = false,titlelocation=:left,yticks=:none)
    annotate!(0,1,label)
    return lp
end
l = @layout [ [la{0.01w} a]  [lb{0.01w} b{0.3h} ; lc{0.01w} c] ];
p = plot(labelplot("A"), p1, labelplot("B"), p2, labelplot("C"), p3, layout = l, size=(800,450));
display(p)
savefig(p, joinpath("results","Figure3.svg"));