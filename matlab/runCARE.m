function [] = runCARE()

overwrite=1;



ensenya('Starting CARE 2.0','Z');

path2data=currentCAREfolder();



id=0;

[~,id]=whosPC();



% care paths

switch id

    case 2 % FLAME 2

        ensenya('Machine not initialized, skipping CARE','r');

    case 3 % FLAME 3

        ensenya('Machine not initialized, skipping CARE','r');

    case 18% Amanda

        ensenya('Machine not initialized, skipping CARE','r');       

    case 420 % Alex

        path2code='C:\Users\austi\Documents\GitHub\BaluLab-CARE\CARE_on_image.py';

        path2mlflow='C:\Users\austi\Documents\SynologyDrive';

    case 422% Belen

        ensenya('Machine not initialized, skipping CARE','r');

end

if(id==0)

    ensenya('Machine not initialized, skipping CARE','r');

else

    extracmd='';

    if(overwrite==1)

        extracmd=' --overwrite';

    end



    cmd1='conda activate CARE';

    cmd2=['python ' path2code ' --data-path ' path2data ' --model-name CARE-1Channel --mlflow-tracking-direc ' path2mlflow extracmd];



    system([cmd1 ' && ' cmd2]);



end



end