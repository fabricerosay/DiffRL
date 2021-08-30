using Flux

const N=3
const NN=N*N

flat(i,j)=N*(j-1)+i

unflat(n)=((n-1)%N+1,div(n-1,N)+1)

function valid(i,j)
    return (1<=i<=N && 1<=j<=N)
end

dir=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

function neighbour(n)
    un=unflat(n)
    nb=Int[]
    for d in dir
        if valid((un.+d)...)
            push!(nb,flat((un.+d)...))
        end
    end
    nb
end

function allneighbour()
    anb=Vector{Vector{Int}}()
    for n in 1:NN
        push!(anb,neighbour(n))
    end
    push!(anb,collect(1:NN))
    return anb
end

mutable struct Transition
    a
end

Flux.@functor Transition

Transition(n::Int)=Transition(Chain(Dense(n,n,relu),Dense(n,n,relu)))

(m::Transition)(x)=m.a(x)
mutable struct Policy
    a
end

Flux.@functor Policy

Policy(n::Int)=Policy(Chain(Dense(2*n,n,relu),Dense(n,1)))

(m::Policy)(x)=m.a(x)

mutable struct Value
    a
end

Flux.@functor Value

Value(n::Int)=Value(Chain(Dense(2*n,n,relu),Dense(n,1,σ)))

(m::Value)(x)=m.a(x)

mutable struct GIN
    transition::Transition
    graph
    ϵ::Float32
    N::Int

end

Flux.@functor GIN

Flux.trainable(m::GIN)=Flux.params(m.transition,m.ϵ)

GIN(n,ϵ)=GIN(Transition(n),allneighbour(),ϵ,NN+1)

function agg(g::GIN,f::AbstractArray{T,2},i) where T
    ag=(1+g.ϵ).*@view f[:,i]
    for n in g.graph[i]
        ag+=@view f[:,n]
    end
    g.transition(ag)
end

function agg(g::GIN,f::AbstractArray{T,3},i) where T
    mapreduce(k->agg(g,f[:,:,k],i),(x,y)->cat(x,y,dims=3),collect(1:size(f)[3]))
end

(g::GIN)(f)=mapreduce(i->agg(g,f,i),(x,y)->cat(x,y,dims=2),collect(1:g.N))

mutable struct Network
    gin
    policy::Policy
    value::Value
end

Flux.@functor Network

Network(n::Int)=Network(Chain(GIN(n,0),GIN(n,0)),Policy(n),Value(n))

function (m::Network)(x)
    b=m.gin[1](x)
    c=m.gin[2](x)
    b=vcat(b,c)
    if length(size(x))==2
        policy=transpose(m.policy(b[:,1:NN]))
    else
        policy=permutedims(m.policy(b[:,1:NN,:]),(2,1,3))
    end
    value=m.value(b[:,NN+1])
    return policy,value
end

function encode(x,f=9)
    L=length(x)
    ans=zeros(f,L+1)
    for (k,el) in enumerate(x)
        ans[:,k].=el
    end
    return ans
end


function alphabeta(game,a,b)
    w,s=winner(game)
    if w
        return game.player*s
    else
        best=-Inf
        for play in 1:9
            if game.board[play]==0
                doMove!(game,play)
                v=alphabeta(game,-b,-a)
                undoMove!(game,play)
                v=-v
                if v>best
                    best=v
                    if best>a
                        a=best
                        if a>=b
                            return a
                        end
                    end
                end
            end
        end
    end
    return a
end

function alphabeta(game)
    lp=legalMove(game)
    best=-Inf
    bestmove=Int[]
    for play in lp
        doMove!(game,play)
        v=-alphabeta(game,-Inf,Inf)
        undoMove!(game,play)
        if v>best
            bestmove=[play]
            best=v
        elseif v==best
            push!(bestmove,play)
        end

    end
    bestmove,best
end

function getProb(bestmove)
    prob=zeros(Float32,9)
    prob[bestmove].=1/length(bestmove)
    prob
end

mutable struct TTT
    board::Vector{Int8}
    player::Int8
    freeMoves::Int8
end

TTT()=TTT(zeros(Int8,9),1,9)

function legalMove(game::TTT)
    lp=Int[]
    for k in 1:9
        if game.board[k]==0
            push!(lp,k)
        end
    end
    lp
end

function doMove!(game,play)
    game.board[play]=game.player
    game.player=-game.player
    game.freeMoves-=1
end

function undoMove!(game,play)
    game.board[play]=0
    game.player=-game.player
    game.freeMoves+=1
end

function winner(game)
    if game.board[1]!=0 && game.board[1]==game.board[2]==game.board[3]
        return true,game.board[1]
    elseif  game.board[4]!=0 && game.board[4]==game.board[5]==game.board[6]
        return true,game.board[4]
    elseif  game.board[7]!=0 && game.board[7]==game.board[8]==game.board[9]
        return true,game.board[7]
    elseif  game.board[1]!=0 && game.board[1]==game.board[4]==game.board[7]
        return true,game.board[1]
    elseif  game.board[2]!=0 && game.board[2]==game.board[5]==game.board[8]
        return true,game.board[2]
    elseif  game.board[3]!=0 && game.board[3]==game.board[6]==game.board[9]
        return true,game.board[3]
    elseif  game.board[1]!=0 && game.board[1]==game.board[5]==game.board[9]
        return true,game.board[1]
    elseif  game.board[7]!=0 && game.board[7]==game.board[5]==game.board[3]
        return true,game.board[7]
    elseif game.freeMoves==0
        return true,Int8(0)
    else
        return false,Int8(0)
    end
end

function (m::Network)(g::TTT)
    p,v=m(encode(g.board*g.player))
    return softmax(p),v
end

function loss(actor,x,y)
    p,v=actor(x)
    L=Flux.Losses.mse(v,y[2])+Flux.Losses.logitcrossentropy(p,y[1])
    return L
end

function getState(g::TTT,t)
    if t==Simplenet
        return Float32.(copy(g.board*g.player))
    elseif t==Network
        return Float32.(encode(g.board*g.player))
    elseif t==ConvNet
        return Float32.(reshape(g.board*g.player,(3,3)))
    elseif t==Doublenet
        an=zeros(Float32,81)
        h=copy(g.board*g.player)
        for k in 1:81
            an[k]=h[(k+div(k-1,9)-1)%9+1]
        end
        #h2=h1[[1,4,7,2,5,8,3,6,9,1,5,9,3,5,7]]
        return an#Float32.(vcat(h,h,h,h,h,h,h,h,h))

    else
        println("type de réseau inconnu")
    end
end

function manygames(n,actor)
    samples=[]
    for i in 1:n
        g=TTT()
        f,r=winner(g)
    while !f
        bestmove,best=alphabeta(g)
        state=getState(g,actor)
        push!(samples,(state,(getProb(bestmove),Float32((1+best)/2))))
        play=rand(legalMove(g))
        doMove!(g,play)
        f,r=winner(g)
    end
end
    samples
end



function mytrain!(loss,ps,data,opt)
    training_loss=0
    ps=Flux.Params(ps)
    faits=10
    L=length(data)
    t0=time()
    for (k,d) in enumerate(data)
        if k/L*100>=faits
            t=time()-t0
            t0=time()
            println("$faits% accomplits, temps:$t")
            faits+=10
        end
        gs=Flux.gradient(ps) do
            training_loss+=loss(d...)
        end
        Flux.update!(opt,ps,gs)
    end
    println("training loss ",training_loss/length(data))

end

function training_loop(actor,n)
    println("generating data")
    samples=manygames(n,typeof(actor))
    println("training...")
    opt=ADAM(0.001)
    mytrain!((x,y)->loss(actor,x,y),Flux.params(actor),samples,opt)
end

mutable struct Simplenet
    base
    policy
    value
end
 Flux.@functor(Simplenet)

 Simplenet()=Simplenet([Dense(9,81,relu),Dense(81,81,relu)],Dense(81,9),Dense(81,1,σ))

 function (m::Simplenet)(x)
    b=m.base[1](x)
    for c in m.base[2:end]
        b=c(b).+b
    end
    return m.policy(b),m.value(b)
end

function (m::Simplenet)(g::TTT)
   p,v=m(g.board*g.player)
   return softmax(p),v
end

mutable struct Doublenet
    base
    policy
    value
end
 Flux.@functor(Doublenet)

 Doublenet()=Doublenet([Dense(81,81,relu),Dense(81,81,relu)],Dense(81,9),Dense(81,1,σ))

 function (m::Doublenet)(x)
    b=m.base[1](x)
    for c in m.base[2:end]
        b=c(b).+b
    end
    return m.policy(b),m.value(b)
end

function (m::Doublenet)(g::TTT)
   p,v=m(getState(g,Doublenet))
   return softmax(p),v
end

mutable struct ConvNet
    base
    policy
    value
end
 Flux.@functor(ConvNet)

 ConvNet()=ConvNet([Dense(3,27,relu),Dense(27,27,relu),Dense(3,3,relu)],Dense(81,9),Dense(81,1,σ))
 function (m::ConvNet)(x)
    b0=m.base[1](x)
    b=transpose(m.base[2](b0))
    b=transpose(m.base[3](b)).+b0
    b=reshape(b,81)
    return m.policy(b),m.value(b)
end

function (m::ConvNet)(g::TTT)
   p,v=m(getState(g,ConvNet))
   return softmax(p),v
end


function pit(actor,actor2,n)
    res=[0,0,0]
    for i in 1:n
        g=TTT()
        f,r=winner(g)
        local play
        while !f
            if g.player==1
                net=actor
            else
                net=actor2
            end
            π,v=net(g)
            lp=legalMove(g)
            π=π[lp]
            π=π/sum(π)
            p=rand()
            s=0
            for (k,el) in enumerate(lp)
                s+=π[k]
                if s>=p
                    play=el
                    break
                end
            end
            doMove!(g,play)
            f,r=winner(g)
        end
        res[r+2]+=1
    end
    return res
end

function fair_pit(net1,net2,n)
    a,b,c=pit(net1,net2,n)
    x,y,z=pit(net2,net1,n)
    v1=(c+x)/(2*n)*100
    d1=(a+z)/(2*n)*100
    n=(b+y)/(2*n)*100
    println("victoires du premier: $v1 %")
    println("nuls: $n %")
    println("victoires du deuxième: $d1 %")
end
