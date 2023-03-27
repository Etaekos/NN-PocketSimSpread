# NN + pocketSimspread
using NamedArrays
using SimSpread
using CUDA


function featurize_stuff(M::NamedArray, α::AbstractFloat, weighted::Bool)
    # Filter matrix
    Mf = copy(M)
    Mf.array = cutoff.(M.array, α, weighted)
    setnames!(Mf, ["f$f" for f in names(Mf, 2)], 2)
    return Mf
end


function get_max_col_names(DP::NamedArray)
    maxcols = Dict()
    rownames = names(DP, 1)
    colnames = names(DP, 2)
    max_col_names = [maxcols[r] = (colnames[findmax(DP[r, :])[2]]) for r in rownames]
    return maxcols
end


function NN(DF::NamedMatrix, Cs::Vector)
    Fs = [f for f in names(DF, 1) if f ∉ Cs]
    mcf = DF[Cs, Fs]

    maxi = get_max_col_names(mcf)

    return maxi
end


function prepare_stuff(DT::T, DF::T, Cs::AbstractVector, Rs::AbstractVector) where {T<:NamedMatrix}
    @assert size(DT, 1) == size(DF, 1) "Different number of compounds!"
    # Get names from matrices
    Fs = [f for f in names(DF, 1) if f ∉ Rs]
    Ds = [d for d in names(DF, 1) if d ∉ Rs]
    Ts = names(DT, 2)
    Fs = ["f$f" for f in Fs]
    #setnames!(DF, ["f$f" for f in names(DF, 2)], 2)
    # Get dimensions of network
    Nc = length(Cs)
    Nf = length(Fs)
    Nd = length(Ds)
    Nt = length(Ts)

    # Construct trilayered graph adjacency matrix
    Mcc = zeros(Nc, Nc)
    Mcd = zeros(Nc, Nd)
    Mcf = DF[Cs, Fs].array
    Mct = zeros(Nc, Nt)

    Mdc = Mcd'
    Mdd = zeros(Nd, Nd)
    Mdf = DF[Ds, Fs].array
    Mdt = DT[Ds, Ts].array

    Mfc = Mcf'
    Mfd = Mdf'
    Mff = zeros(Nf, Nf)
    Mft = zeros(Nf, Nt)

    Mtc = Mct'
    Mtd = Mdt'
    Mtf = Mft'
    Mtt = zeros(Nt, Nt)

    A = Matrix(
        [Mcc Mcd Mcf Mct
         Mdc Mdd Mdf Mdt
         Mfc Mfd Mff Mft
         Mtc Mtd Mtf Mtt]
    )
    Cs = ["c$f" for f in Cs]
    namedA = NamedArray(A, (vcat(Cs, Ds, Fs, Ts), vcat(Cs, Ds, Fs, Ts)))
    namedB = deepcopy(namedA)
    namedB[Cs, :] .= 0
    namedB[:, Cs] .= 0

    return namedA, namedB
end



function save_stuff(filepath::String, fidx::Int64, C::AbstractVector, R::NamedMatrix, DT::NamedMatrix, dicto::Dict)
    open(filepath, "a+") do io
        for c in C, t in names(DT, 2)
            a = [k for (k,v) in dicto if v==c]
            a = (a[1])
            write(io, "$fidx, \"$a\", \"$t\", $(R[c,t]), $(DT[a,t])\n")
        end
    end
end

route = pwd()
DT = read_namedmatrix(route*"/biolip_bioactive_definitive_1ligpdb_tenfold_LigTrg_matrix.txt")
DF = read_namedmatrix(route*"/bioactive_defintive_1ligpdb_SrelSquare_ligindex.txt")
DP = read_namedmatrix(route*"/bioactive_definitive_1ligpdb_tenfold.txt.tanmat")

groups = split(DT, 10)
atc = Array(0.0:0.1:1.0)

for αₜ in atc
    for group in 1:length(groups)
        f_srel = featurize_stuff(DF, αₜ, false)
        e = groups[group]
        E = NN(DP,e)
        NNs = unique(collect(values(E)))
        (A,B) = prepare_stuff(DT, f_srel, NNs, e)
        R = SimSpread.predict((A,B), DT, GPU=true)
        #clean!(R, A, DT)
        save_stuff(route*"/alt_outs/NNPocketSimSpread_Bioactive1ligpdb_Tenfold_ecfp4_TcNa_Srel$(αₜ)_seed1.out", group, NNs, R, DT, E)
    end
end
